# TrainIt Setup Assistant (vNext I0, ADR-0027): open the project's bound Isaac stage and PLAY it.
#
# Run INSIDE Isaac Sim (Kit), never with a plain python:
#     TRAINIT_STAGE=/path/to/cell.usd  <isaac-sim>/isaac-sim.sh --exec open_and_play.py
# TSA's launcher runs exactly that line for "Start Isaac" (the stage is the project's
# deployment.isaac.stage). The stage file is opened, never modified or saved.
#
# Why the waits (measured on a reference cell, vNext M3-B):
# - opening the stage and pressing PLAY before "Isaac Sim Full App is loaded" leaves the stage's
#   ROS graphs silent: wait for the app's own startup first;
# - the first play after a scripted stage open left the OnPlaybackTick graphs unevaluated: a
#   play / stop / play cycle (what the toolbar does) arms them;
# - PLAY must precede the ROS bring-up: a controller started before it commands the zero pose.
import asyncio
import os

import omni.kit.app
import omni.timeline
import omni.usd

STAGE = os.environ.get('TRAINIT_STAGE', '')


async def _frames(n):
    app = omni.kit.app.get_app()
    for _ in range(n):
        await app.next_update_async()


async def _main():
    if not STAGE:
        print('TRAINIT: TRAINIT_STAGE is not set: no stage to open', flush=True)
        return
    app = omni.kit.app.get_app()
    ready = getattr(app, 'is_app_ready', None)
    while ready is not None and not ready():
        await app.next_update_async()
    await _frames(600)
    ok, err = await omni.usd.get_context().open_stage_async(STAGE)
    print(f'TRAINIT: open_stage {STAGE} ok={ok} err={err}', flush=True)
    if not ok:
        return
    await _frames(600)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    await _frames(300)
    timeline.stop()
    await _frames(300)
    timeline.play()
    await _frames(120)
    print(f'TRAINIT: PLAYING is_playing={timeline.is_playing()}', flush=True)
    while True:                       # report only: a stopped timeline is never restarted silently
        await _frames(1200)
        print(f'TRAINIT: is_playing={timeline.is_playing()}', flush=True)


asyncio.ensure_future(_main())
