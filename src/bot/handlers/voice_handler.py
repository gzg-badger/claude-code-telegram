"""Voice note handler for BB3K Telegram bot.

Downloads Telegram voice notes (.ogg), transcribes via faster-whisper,
shows the transcription to George, then sends it to Claude via the orchestrator.
"""

import asyncio
import os
import tempfile

import structlog
from telegram import Update
from telegram.ext import ContextTypes

logger = structlog.get_logger()

# Lazy-loaded model
_model = None
_MODEL_SIZE = os.environ.get("BB3K_WHISPER_MODEL", "base")


def _get_model():
    """Lazy-load the faster-whisper model on first use."""
    global _model
    if _model is None:
        logger.info("Loading faster-whisper model", model_size=_MODEL_SIZE)
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            _MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisper model loaded")
    return _model


def transcribe_audio(ogg_path: str) -> str:
    """Transcribe an audio file and return the text."""
    model = _get_model()

    segments, info = model.transcribe(
        ogg_path,
        language="en",
        beam_size=5,
        vad_filter=True,
    )

    text = " ".join(segment.text.strip() for segment in segments)
    text = text.strip()

    if not text:
        return ""

    logger.info(
        "Transcribed audio",
        duration=f"{info.duration:.1f}s",
        chars=len(text),
        language=info.language,
        probability=f"{info.language_probability:.2f}",
    )
    return text


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice notes: download, transcribe, send to Claude directly.

    Mirrors the orchestrator's agentic_text method for Claude interaction,
    with transcription prepended.
    """
    user_id = update.effective_user.id

    # Check voice duration (max 5 minutes)
    voice = update.message.voice
    if voice.duration and voice.duration > 300:
        await update.message.reply_text("Voice note too long (max 5 minutes).")
        return

    logger.info("Processing voice note", user_id=user_id, duration=voice.duration)

    # Download voice file
    file = await context.bot.get_file(voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        await file.download_to_drive(tmp_path)

        # Transcribe
        await update.effective_chat.send_action("typing")

        loop = asyncio.get_running_loop()
        transcription = await loop.run_in_executor(None, transcribe_audio, tmp_path)

        if not transcription:
            await update.message.reply_text("[No speech detected]")
            return

        # Show transcription to George
        await update.message.reply_text(
            f"\U0001f399\ufe0f {transcription}",
            reply_to_message_id=update.message.message_id,
        )

        # Check for quick commands in transcription
        from .quick_commands import route_command

        quick_result = route_command(transcription)
        if quick_result is not None:
            await update.message.reply_text(quick_result)
            return

        # --- Claude interaction (mirrors orchestrator.agentic_text) ---

        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await update.message.reply_text(
                "Claude integration not available. Check configuration."
            )
            return

        settings = context.bot_data.get("settings")
        current_dir = context.user_data.get(
            "current_directory", settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")
        force_new = bool(context.user_data.get("force_new_session"))

        progress_msg = await update.message.reply_text("Working...")

        try:
            claude_response = await claude_integration.run_command(
                prompt=transcription,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                force_new=force_new,
            )

            # Clear one-shot flag after successful run
            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id

            # Format response (same as orchestrator)
            from ..utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

        except Exception as e:
            logger.error("Claude integration failed", error=str(e), user_id=user_id)
            from ..utils.formatting import FormattedMessage

            formatted_messages = [
                FormattedMessage(f"Claude error: {str(e)[:200]}", parse_mode=None)
            ]

        await progress_msg.delete()

        # Send formatted messages with redaction + HTML fallback
        from ...security.redaction import redact_secrets

        for i, message in enumerate(formatted_messages):
            message.text = redact_secrets(message.text)
            try:
                await update.message.reply_text(
                    message.text,
                    parse_mode=message.parse_mode,
                    reply_to_message_id=(
                        update.message.message_id if i == 0 else None
                    ),
                )
            except Exception as e:
                logger.warning(
                    "Failed to send HTML response, retrying as plain text",
                    error=str(e),
                    message_index=i,
                )
                try:
                    await update.message.reply_text(
                        message.text,
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                except Exception:
                    await update.message.reply_text(
                        "Failed to send response. Please try again.",
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
            if i < len(formatted_messages) - 1:
                await asyncio.sleep(0.5)

    except ImportError:
        logger.error("faster-whisper not installed")
        await update.message.reply_text(
            "Voice transcription unavailable - faster-whisper not installed on VPS."
        )
    except Exception as e:
        logger.exception("Voice processing failed", error=str(e))
        await update.message.reply_text(f"Voice processing failed: {str(e)[:100]}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
