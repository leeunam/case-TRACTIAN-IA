from __future__ import annotations

import asyncio
import sys

if sys.platform != "win32":
    import uvloop

    # Alguns hosts não despertam de modo confiável o selector asyncio padrão
    # após callbacks de threads. O agente usa aiosqlite nos testes, então o
    # backend uvloop evita deadlock sem alterar o runtime do pacote.
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
