"""J.A.R.V.I.S. - run the assistant.

    python main.py              voice mode: say "Jarvis" to wake it up
    python main.py --text       type instead of talking (no microphone needed)
    python main.py --text --speak   type, and hear the replies
    python main.py --doctor     check the set-up
    python main.py --list-mics  show microphones (set MIC_DEVICE_INDEX in .env)
    python main.py --debug      show the detected intent and confidence for every command
"""

from __future__ import annotations

import argparse
import logging
import sys

from jarvis.config import Settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. personal assistant")
    parser.add_argument("--text", action="store_true", help="type commands instead of speaking")
    parser.add_argument("--speak", action="store_true", help="with --text: also speak the replies")
    parser.add_argument("--doctor", action="store_true", help="check your set-up and exit")
    parser.add_argument("--list-mics", action="store_true", help="list microphones and exit")
    parser.add_argument("--debug", action="store_true", help="show intents, confidences and logs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("urllib3", "filelock", "httpx", "httpcore", "matplotlib", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = Settings.from_env()

    if args.doctor:
        from jarvis.doctor import main as doctor

        return doctor(settings)

    if args.list_mics:
        from jarvis.speech.listener import Listener, ListenerError

        try:
            microphones = Listener.list_microphones()
        except ListenerError as exc:
            print(exc)
            return 1
        print("Input devices (put the number in .env as MIC_DEVICE_INDEX):")
        for index, name in microphones:
            print(f"{index:3d}  {name}")
        return 0

    from jarvis.app import TextApp, VoiceApp, start_scheduler
    from jarvis.assistant import Assistant
    from jarvis.speech.speaker import Speaker

    print("Starting J.A.R.V.I.S. ...")
    assistant = Assistant.build(settings)
    scheduler = start_scheduler(assistant)
    speaker = None
    try:
        if args.text:
            speaker = Speaker(settings.tts_engine, settings.tts_voice, settings.tts_rate) if args.speak else None
            TextApp(assistant, speaker, debug=args.debug).run()
        else:
            from jarvis.speech.listener import Listener, ListenerError

            try:
                listener = Listener(settings.stt_engine, settings.effective_wake_engine, settings.language,
                                    settings.vosk_model_path, settings.mic_device_index, settings.wake_word)
            except ListenerError as exc:
                print(f"Voice mode is not available: {exc}\nTry `python main.py --text`, or run "
                      f"`python main.py --doctor` to see what's missing.")
                return 1
            speaker = Speaker(settings.tts_engine, settings.tts_voice, settings.tts_rate)
            VoiceApp(assistant, listener, speaker, settings, debug=args.debug).run()
    finally:
        scheduler.stop()
        if speaker:
            speaker.close()
        assistant.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
