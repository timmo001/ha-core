"""AI tasks to be handled by agents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import io
import mimetypes
from pathlib import Path
import tempfile
from typing import Any

import voluptuous as vol

from homeassistant.components import camera, conversation, image, media_source
from homeassistant.components.http.auth import async_sign_path
from homeassistant.core import HomeAssistant, ServiceResponse, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.chat_session import ChatSession, async_get_chat_session
from homeassistant.util import RE_SANITIZE_FILENAME, slugify

from .const import (
    DATA_COMPONENT,
    DATA_MEDIA_SOURCE,
    DATA_PREFERENCES,
    DOMAIN,
    IMAGE_DIR,
    IMAGE_EXPIRY_TIME,
    AITaskEntityFeature,
)


def _save_camera_snapshot(image_data: camera.Image | image.Image) -> Path:
    """Save camera snapshot to temp file."""
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=mimetypes.guess_extension(image_data.content_type, False),
        delete=False,
    ) as temp_file:
        temp_file.write(image_data.content)
        return Path(temp_file.name)


async def _resolve_attachments(
    hass: HomeAssistant,
    session: ChatSession,
    attachments: list[dict] | None = None,
) -> list[conversation.Attachment]:
    """Resolve attachments for a task."""
    resolved_attachments: list[conversation.Attachment] = []
    created_files: list[Path] = []

    for attachment in attachments or []:
        media_content_id = attachment["media_content_id"]

        # Special case for certain media sources
        for integration in camera, image:
            media_source_prefix = f"media-source://{integration.DOMAIN}/"
            if not media_content_id.startswith(media_source_prefix):
                continue

            # Extract entity_id from the media content ID
            entity_id = media_content_id.removeprefix(media_source_prefix)

            # Get snapshot from entity
            image_data = await integration.async_get_image(hass, entity_id)

            temp_filename = await hass.async_add_executor_job(
                _save_camera_snapshot, image_data
            )
            created_files.append(temp_filename)

            resolved_attachments.append(
                conversation.Attachment(
                    media_content_id=media_content_id,
                    mime_type=image_data.content_type,
                    path=temp_filename,
                )
            )
            break
        else:
            # Handle regular media sources
            media = await media_source.async_resolve_media(hass, media_content_id, None)
            if media.path is None:
                raise HomeAssistantError(
                    "Only local attachments are currently supported"
                )
            resolved_attachments.append(
                conversation.Attachment(
                    media_content_id=media_content_id,
                    mime_type=media.mime_type,
                    path=media.path,
                )
            )

    if not created_files:
        return resolved_attachments

    def cleanup_files() -> None:
        """Cleanup temporary files."""
        for file in created_files:
            file.unlink(missing_ok=True)

    @callback
    def cleanup_files_callback() -> None:
        """Cleanup temporary files."""
        hass.async_add_executor_job(cleanup_files)

    session.async_on_cleanup(cleanup_files_callback)

    return resolved_attachments


async def async_generate_data(
    hass: HomeAssistant,
    *,
    task_name: str,
    entity_id: str | None = None,
    instructions: str,
    structure: vol.Schema | None = None,
    attachments: list[dict] | None = None,
    llm_api: llm.API | None = None,
) -> GenDataTaskResult:
    """Run a data generation task in the AI Task integration."""
    if entity_id is None:
        entity_id = hass.data[DATA_PREFERENCES].gen_data_entity_id

    if entity_id is None:
        raise HomeAssistantError("No entity_id provided and no preferred entity set")

    entity = hass.data[DATA_COMPONENT].get_entity(entity_id)
    if entity is None:
        raise HomeAssistantError(f"AI Task entity {entity_id} not found")

    if AITaskEntityFeature.GENERATE_DATA not in entity.supported_features:
        raise HomeAssistantError(
            f"AI Task entity {entity_id} does not support generating data"
        )

    if (
        attachments
        and AITaskEntityFeature.SUPPORT_ATTACHMENTS not in entity.supported_features
    ):
        raise HomeAssistantError(
            f"AI Task entity {entity_id} does not support attachments"
        )

    with async_get_chat_session(hass) as session:
        resolved_attachments = await _resolve_attachments(hass, session, attachments)

        return await entity.internal_async_generate_data(
            session,
            GenDataTask(
                name=task_name,
                instructions=instructions,
                structure=structure,
                attachments=resolved_attachments or None,
                llm_api=llm_api,
            ),
        )


async def async_generate_image(
    hass: HomeAssistant,
    *,
    task_name: str,
    entity_id: str | None = None,
    instructions: str,
    attachments: list[dict] | None = None,
) -> ServiceResponse:
    """Run an image generation task in the AI Task integration."""
    if entity_id is None:
        entity_id = hass.data[DATA_PREFERENCES].gen_image_entity_id

    if entity_id is None:
        raise HomeAssistantError("No entity_id provided and no preferred entity set")

    entity = hass.data[DATA_COMPONENT].get_entity(entity_id)
    if entity is None:
        raise HomeAssistantError(f"AI Task entity {entity_id} not found")

    if AITaskEntityFeature.GENERATE_IMAGE not in entity.supported_features:
        raise HomeAssistantError(
            f"AI Task entity {entity_id} does not support generating images"
        )

    if (
        attachments
        and AITaskEntityFeature.SUPPORT_ATTACHMENTS not in entity.supported_features
    ):
        raise HomeAssistantError(
            f"AI Task entity {entity_id} does not support attachments"
        )

    with async_get_chat_session(hass) as session:
        resolved_attachments = await _resolve_attachments(hass, session, attachments)

        task_result = await entity.internal_async_generate_image(
            session,
            GenImageTask(
                name=task_name,
                instructions=instructions,
                attachments=resolved_attachments or None,
            ),
        )

    service_result = task_result.as_dict()
    image_data = service_result.pop("image_data")
    if service_result.get("revised_prompt") is None:
        service_result["revised_prompt"] = instructions

    source = hass.data[DATA_MEDIA_SOURCE]

    current_time = datetime.now()
    ext = mimetypes.guess_extension(task_result.mime_type, False) or ".png"
    sanitized_task_name = RE_SANITIZE_FILENAME.sub("", slugify(task_name))

    image_file = ImageData(
        filename=f"{current_time.strftime('%Y-%m-%d_%H%M%S')}_{sanitized_task_name}{ext}",
        file=io.BytesIO(image_data),
        content_type=task_result.mime_type,
    )

    target_folder = media_source.MediaSourceItem.from_uri(
        hass, f"media-source://{DOMAIN}/{IMAGE_DIR}", None
    )

    service_result["media_source_id"] = await source.async_upload_media(
        target_folder, image_file
    )

    item = media_source.MediaSourceItem.from_uri(
        hass, service_result["media_source_id"], None
    )
    service_result["url"] = async_sign_path(
        hass,
        (await source.async_resolve_media(item)).url,
        timedelta(seconds=IMAGE_EXPIRY_TIME),
    )

    return service_result


@dataclass(slots=True)
class GenDataTask:
    """Gen data task to be processed."""

    name: str
    """Name of the task."""

    instructions: str
    """Instructions on what needs to be done."""

    structure: vol.Schema | None = None
    """Optional structure for the data to be generated."""

    attachments: list[conversation.Attachment] | None = None
    """List of attachments to go along the instructions."""

    llm_api: llm.API | None = None
    """API to provide to the LLM."""

    def __str__(self) -> str:
        """Return task as a string."""
        return f"<GenDataTask {self.name}: {id(self)}>"


@dataclass(slots=True)
class GenDataTaskResult:
    """Result of gen data task."""

    conversation_id: str
    """Unique identifier for the conversation."""

    data: Any
    """Data generated by the task."""

    def as_dict(self) -> dict[str, Any]:
        """Return result as a dict."""
        return {
            "conversation_id": self.conversation_id,
            "data": self.data,
        }


@dataclass(slots=True)
class GenImageTask:
    """Gen image task to be processed."""

    name: str
    """Name of the task."""

    instructions: str
    """Instructions on what needs to be done."""

    attachments: list[conversation.Attachment] | None = None
    """List of attachments to go along the instructions."""

    def __str__(self) -> str:
        """Return task as a string."""
        return f"<GenImageTask {self.name}: {id(self)}>"


@dataclass(slots=True)
class GenImageTaskResult:
    """Result of gen image task."""

    image_data: bytes
    """Raw image data generated by the model."""

    conversation_id: str
    """Unique identifier for the conversation."""

    mime_type: str
    """MIME type of the generated image."""

    width: int | None = None
    """Width of the generated image, if available."""

    height: int | None = None
    """Height of the generated image, if available."""

    model: str | None = None
    """Model used to generate the image, if available."""

    revised_prompt: str | None = None
    """Revised prompt used to generate the image, if applicable."""

    def as_dict(self) -> dict[str, Any]:
        """Return result as a dict."""
        return {
            "image_data": self.image_data,
            "conversation_id": self.conversation_id,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "model": self.model,
            "revised_prompt": self.revised_prompt,
        }


@dataclass(slots=True)
class ImageData:
    """Implementation of media_source.local_source.UploadedFile protocol."""

    filename: str
    file: io.IOBase
    content_type: str


@dataclass(slots=True)
class GenThemeTask:
    """Gen theme task to be processed."""

    instructions: str
    """Instructions describing the desired theme aesthetic."""

    def __str__(self) -> str:
        """Return task as a string."""
        return f"<GenThemeTask: {id(self)}>"


@dataclass(slots=True)
class GenThemeTaskResult:
    """Result of gen theme task."""

    conversation_id: str
    """Unique identifier for the conversation."""

    name: str
    """AI-generated name for the theme."""

    variables: dict[str, str]
    """Theme variables that apply to all modes."""

    light: dict[str, str]
    """Light mode specific variable overrides."""

    dark: dict[str, str]
    """Dark mode specific variable overrides."""

    def as_dict(self) -> dict[str, Any]:
        """Return result as a dict for service response.

        Returns the theme ready to use in config/themes/x.yaml.
        """
        # Build theme content with shared variables at top level
        theme_content: dict[str, Any] = dict(self.variables)

        # Add modes if there are mode-specific overrides
        if self.light or self.dark:
            modes: dict[str, dict[str, str]] = {}
            if self.light:
                modes["light"] = self.light
            if self.dark:
                modes["dark"] = self.dark
            theme_content["modes"] = modes

        return {
            self.name: theme_content,
        }


# Theme variable schema for AI to generate
# Based on Home Assistant's frontend THEME_SCHEMA and common variables from
# popular themes like Catppuccin. Explicit variable names guide the AI on
# exactly what variables to generate.

# Mode-specific variables (used for both light and dark modes)
_THEME_MODE_SCHEMA = {
    vol.Optional("primary-background-color"): str,
    vol.Optional("secondary-background-color"): str,
    vol.Optional("card-background-color"): str,
    vol.Optional("primary-text-color"): str,
    vol.Optional("secondary-text-color"): str,
    vol.Optional("text-primary-color"): str,
    vol.Optional("disabled-text-color"): str,
    vol.Optional("divider-color"): str,
    vol.Optional("outline-color"): str,
    vol.Optional("state-icon-color"): str,
    vol.Optional("sidebar-background-color"): str,
    vol.Optional("sidebar-text-color"): str,
    vol.Optional("sidebar-icon-color"): str,
    vol.Optional("sidebar-selected-icon-color"): str,
    vol.Optional("sidebar-selected-text-color"): str,
    vol.Optional("app-header-background-color"): str,
    vol.Optional("app-header-text-color"): str,
    vol.Optional("input-fill-color"): str,
    vol.Optional("input-ink-color"): str,
    vol.Optional("input-label-ink-color"): str,
    vol.Optional("switch-checked-color"): str,
    vol.Optional("switch-unchecked-button-color"): str,
    vol.Optional("switch-unchecked-track-color"): str,
    vol.Optional("slider-color"): str,
    vol.Optional("scrollbar-thumb-color"): str,
    vol.Optional("disabled-color"): str,
    vol.Optional("state-active-color"): str,
    vol.Optional("state-inactive-color"): str,
    # Energy colors
    vol.Optional("energy-grid-consumption-color"): str,
    vol.Optional("energy-grid-return-color"): str,
    vol.Optional("energy-solar-color"): str,
    vol.Optional("energy-battery-out-color"): str,
    vol.Optional("energy-battery-in-color"): str,
    vol.Optional("energy-gas-color"): str,
    vol.Optional("energy-water-color"): str,
    # CodeMirror colors
    vol.Optional("codemirror-keyword"): str,
    vol.Optional("codemirror-operator"): str,
    vol.Optional("codemirror-variable"): str,
    vol.Optional("codemirror-string"): str,
    vol.Optional("codemirror-comment"): str,
    vol.Optional("codemirror-number"): str,
}

THEME_STRUCTURE_SCHEMA = vol.Schema(
    {
        # Theme metadata
        vol.Required("name"): str,
        # Core interface colors (shared between modes)
        vol.Optional("primary-color"): str,
        vol.Optional("accent-color"): str,
        vol.Optional("error-color"): str,
        vol.Optional("warning-color"): str,
        vol.Optional("success-color"): str,
        vol.Optional("info-color"): str,
        # Named colors for states and charts
        vol.Optional("red-color"): str,
        vol.Optional("pink-color"): str,
        vol.Optional("purple-color"): str,
        vol.Optional("deep-purple-color"): str,
        vol.Optional("indigo-color"): str,
        vol.Optional("blue-color"): str,
        vol.Optional("light-blue-color"): str,
        vol.Optional("cyan-color"): str,
        vol.Optional("teal-color"): str,
        vol.Optional("green-color"): str,
        vol.Optional("light-green-color"): str,
        vol.Optional("lime-color"): str,
        vol.Optional("yellow-color"): str,
        vol.Optional("amber-color"): str,
        vol.Optional("orange-color"): str,
        vol.Optional("deep-orange-color"): str,
        vol.Optional("brown-color"): str,
        vol.Optional("grey-color"): str,
        vol.Optional("light-grey-color"): str,
        vol.Optional("dark-grey-color"): str,
        # Light and dark mode specific variables
        vol.Optional("light"): _THEME_MODE_SCHEMA,
        vol.Optional("dark"): _THEME_MODE_SCHEMA,
    }
)

# System prompt providing context about HA themes and default values
THEME_SYSTEM_PROMPT = """You are a Home Assistant theme designer. Generate CSS custom property values for a Home Assistant theme.

## Default Theme Values (Light Mode)
- primary-color: rgb(0, 154, 199) (Home Assistant blue)
- accent-color: rgb(255, 152, 0) (orange)
- primary-background-color: rgb(250, 250, 250)
- secondary-background-color: rgb(229, 229, 229)
- card-background-color: rgb(255, 255, 255)
- primary-text-color: rgb(33, 33, 33)
- secondary-text-color: rgb(114, 114, 114)
- disabled-text-color: rgb(189, 189, 189)
- divider-color: rgba(0, 0, 0, 0.12)
- app-header-background-color: rgb(0, 154, 199)
- sidebar-background-color: rgb(255, 255, 255)

## Default Theme Values (Dark Mode)
- primary-background-color: rgb(17, 17, 17)
- secondary-background-color: rgb(40, 40, 40)
- card-background-color: rgb(28, 28, 28)
- primary-text-color: rgb(225, 225, 225)
- secondary-text-color: rgb(155, 155, 155)
- disabled-text-color: rgb(111, 111, 111)
- app-header-background-color: rgb(16, 30, 36)

## Color Format Guidelines
1. **Preferred**: Use `rgb(r, g, b)` or `rgba(r, g, b, a)` for most colors
2. **For transparency**: Use `rgba()` with alpha values (e.g., `rgba(0, 0, 0, 0.12)`)
3. **For advanced/HDR themes**: You may use modern CSS color functions:
   - `oklch(L C H)` - Perceptually uniform, HDR-capable (e.g., `oklch(0.7 0.15 180)`)
   - `oklab(L a b)` - Perceptually uniform lab space
   - `color(display-p3 r g b)` - Wide gamut P3 color space
   - `lch(L C H)` - CIE LCH color space
4. **Fallback hex**: Use hex only when rgb would be redundant (e.g., pure white `#fff`)

## Instructions
1. Create a cohesive color palette based on the user's description
2. Only include variables that differ from the defaults
3. Ensure good contrast between text and background colors (WCAG 2.1 AA minimum)
4. Generate a creative, descriptive name for the theme (2-4 words)
5. The "light" and "dark" objects contain mode-specific overrides
6. Shared colors (like primary-color, accent-color, named colors) go at the top level
7. For vibrant/HDR themes, consider using oklch() for more saturated colors that work on HDR displays

Focus on creating a visually appealing, accessible theme that matches the user's description."""


async def async_generate_theme(
    hass: HomeAssistant,
    *,
    entity_id: str | None = None,
    instructions: str,
) -> GenThemeTaskResult:
    """Run a theme generation task using the AI Task integration."""
    if entity_id is None:
        entity_id = hass.data[DATA_PREFERENCES].gen_data_entity_id

    if entity_id is None:
        raise HomeAssistantError("No entity_id provided and no preferred entity set")

    entity = hass.data[DATA_COMPONENT].get_entity(entity_id)
    if entity is None:
        raise HomeAssistantError(f"AI Task entity {entity_id} not found")

    if AITaskEntityFeature.GENERATE_DATA not in entity.supported_features:
        raise HomeAssistantError(
            f"AI Task entity {entity_id} does not support generating data"
        )

    # Build the full instructions with system context
    full_instructions = (
        f"{THEME_SYSTEM_PROMPT}\n\n"
        f"## User Request\n{instructions}\n\n"
        "Generate a theme based on this description."
    )

    with async_get_chat_session(hass) as session:
        result = await entity.internal_async_generate_data(
            session,
            GenDataTask(
                name="generate_theme",
                instructions=full_instructions,
                structure=THEME_STRUCTURE_SCHEMA,
                attachments=None,
                llm_api=None,
            ),
        )

    # Extract theme data from the result
    theme_data = result.data

    # Separate shared variables from mode-specific ones
    variables: dict[str, str] = {}
    light_vars: dict[str, str] = {}
    dark_vars: dict[str, str] = {}

    for key, value in theme_data.items():
        if key == "name":
            continue
        if key == "light":
            light_vars = {k: v for k, v in value.items() if v}
        elif key == "dark":
            dark_vars = {k: v for k, v in value.items() if v}
        elif value:  # Only include non-empty values
            variables[key] = value

    return GenThemeTaskResult(
        conversation_id=result.conversation_id,
        name=theme_data.get("name", "Generated Theme"),
        variables=variables,
        light=light_vars,
        dark=dark_vars,
    )
