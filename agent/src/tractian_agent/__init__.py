"""Inicialização segura do pacote antes que qualquer modelo Pydantic exista."""

from __future__ import annotations

import os


_DISABLED_PLUGINS_ENV = "PYDANTIC_DISABLE_PLUGINS"
_LOGFIRE_PLUGIN = "logfire-plugin"
_DISABLE_ALL_SENTINELS = frozenset({"1", "true", "__all__"})


def _disable_automatic_logfire_pydantic_plugin() -> None:
    current = os.environ.get(_DISABLED_PLUGINS_ENV)
    if current in _DISABLE_ALL_SENTINELS:
        return
    names = [] if current is None else current.split(",")
    if _LOGFIRE_PLUGIN not in names:
        names.append(_LOGFIRE_PLUGIN)
    os.environ[_DISABLED_PLUGINS_ENV] = ",".join(name for name in names if name)


_disable_automatic_logfire_pydantic_plugin()
