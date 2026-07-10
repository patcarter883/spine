"""Provider fallback on endpoint outage.

A provider entry names a standby via ``fallback_provider``; at model-build
time a TTL-cached TCP health check on the primary decides whether to build
the primary or the standby. Motivation: the remote mini-sglang serve crashes
under concurrent large-prompt load (probe 19, run 0fa536c2) — runs should
degrade to the local Lemonade endpoint instead of tripping the circuit
breaker, and return to the primary automatically once it is back.
"""

from unittest.mock import patch

import pytest

import spine.agents.helpers as helpers

PRIMARY = {
    "name": "remote-qwen",
    "model": "openai:remote/Qwen-Big",
    "base_url": "http://10.0.0.1:1919/v1",
    "fallback_provider": "local-standby",
}
STANDBY = {
    "name": "local-standby",
    "model": "openai:Qwen-Local-GGUF",
    "base_url": "http://localhost:8010/v1",
    "enabled": False,
}


@pytest.fixture(autouse=True)
def _clear_health_cache():
    helpers._ENDPOINT_HEALTH.clear()
    yield
    helpers._ENDPOINT_HEALTH.clear()


def _with_lookup(monkeypatch, providers):
    class FakeConfig:
        def _lookup_provider_by_name(self, name):
            return next((p for p in providers if p.get("name") == name), None)

    monkeypatch.setattr(
        "spine.config.SpineConfig.load", staticmethod(lambda: FakeConfig())
    )


class TestApplyProviderFallback:
    def test_healthy_primary_untouched(self, monkeypatch):
        with patch.object(helpers, "_endpoint_healthy", return_value=True) as hc:
            cfg, spec = helpers._apply_provider_fallback(dict(PRIMARY), PRIMARY["model"])
        assert cfg["name"] == "remote-qwen"
        assert spec == "openai:remote/Qwen-Big"
        hc.assert_called_once_with("http://10.0.0.1:1919/v1", "openai:remote/Qwen-Big")

    def test_dead_primary_reroutes_to_standby(self, monkeypatch):
        _with_lookup(monkeypatch, [PRIMARY, STANDBY])
        health = {"http://10.0.0.1:1919/v1": False, "http://localhost:8010/v1": True}
        with patch.object(
            helpers, "_endpoint_healthy",
            side_effect=lambda url, model=None: health[url],
        ):
            cfg, spec = helpers._apply_provider_fallback(dict(PRIMARY), PRIMARY["model"])
        assert cfg["name"] == "local-standby"
        assert spec == "openai:Qwen-Local-GGUF"

    def test_both_dead_stays_on_primary(self, monkeypatch):
        _with_lookup(monkeypatch, [PRIMARY, STANDBY])
        with patch.object(helpers, "_endpoint_healthy", return_value=False):
            cfg, spec = helpers._apply_provider_fallback(dict(PRIMARY), PRIMARY["model"])
        assert cfg["name"] == "remote-qwen"

    def test_missing_fallback_entry_stays_on_primary(self, monkeypatch):
        _with_lookup(monkeypatch, [PRIMARY])
        with patch.object(helpers, "_endpoint_healthy", return_value=False):
            cfg, spec = helpers._apply_provider_fallback(dict(PRIMARY), PRIMARY["model"])
        assert cfg["name"] == "remote-qwen"

    def test_non_local_fallback_rejected(self, monkeypatch):
        openrouter_standby = {
            "name": "local-standby",
            "model": "openrouter:some/model",
            "base_url": "http://localhost:8010/v1",
        }
        _with_lookup(monkeypatch, [PRIMARY, openrouter_standby])
        with patch.object(helpers, "_endpoint_healthy", return_value=False):
            cfg, spec = helpers._apply_provider_fallback(dict(PRIMARY), PRIMARY["model"])
        assert cfg["name"] == "remote-qwen"
        assert spec == "openai:remote/Qwen-Big"

    def test_cycle_guard_terminates(self, monkeypatch):
        a = dict(PRIMARY, fallback_provider="b", name="a")
        b = {
            "name": "b",
            "model": "openai:B",
            "base_url": "http://b:1/v1",
            "fallback_provider": "a",
        }
        _with_lookup(monkeypatch, [a, b])
        with patch.object(helpers, "_endpoint_healthy", return_value=False):
            cfg, spec = helpers._apply_provider_fallback(dict(a), a["model"])
        assert cfg["name"] == "a"

    def test_no_fallback_key_short_circuits(self):
        plain = {"name": "p", "model": "openai:P", "base_url": "http://p:1/v1"}
        with patch.object(helpers, "_endpoint_healthy") as hc:
            cfg, spec = helpers._apply_provider_fallback(dict(plain), plain["model"])
        hc.assert_not_called()
        assert cfg["name"] == "p"


class TestEndpointHealthy:
    def test_ttl_cache_hit_skips_probe(self, monkeypatch):
        import socket as socket_mod

        calls = []

        def fake_create_connection(addr, timeout):
            calls.append(addr)
            raise OSError("refused")

        monkeypatch.setattr(socket_mod, "create_connection", fake_create_connection)
        assert helpers._endpoint_healthy("http://10.0.0.9:1919/v1") is False
        assert helpers._endpoint_healthy("http://10.0.0.9:1919/v1") is False
        assert len(calls) == 1  # second call served from cache

    def test_connect_success_is_healthy(self, monkeypatch):
        import socket as socket_mod

        class FakeSock:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            socket_mod, "create_connection", lambda addr, timeout: FakeSock()
        )
        assert helpers._endpoint_healthy("http://10.0.0.9:1919/v1") is True


class TestResolveModelFallbackIntegration:
    def test_resolve_model_builds_standby_when_primary_down(self, monkeypatch):
        _with_lookup(monkeypatch, [PRIMARY, STANDBY])
        monkeypatch.setattr(
            helpers, "_active_provider_config", lambda **kw: dict(PRIMARY)
        )
        monkeypatch.setattr(
            helpers,
            "_model_spec_from_config",
            lambda config, phase=None, escalation_level=0: PRIMARY["model"],
        )
        health = {"http://10.0.0.1:1919/v1": False, "http://localhost:8010/v1": True}
        built = {}
        health_fn = lambda url, model=None: health[url]  # noqa: E731

        def fake_build(model_spec, provider_cfg):
            built["spec"] = model_spec
            built["cfg"] = provider_cfg
            return "model-instance"

        monkeypatch.setattr(helpers, "_build_local_model", fake_build)
        with patch.object(helpers, "_endpoint_healthy", side_effect=health_fn):
            result = helpers.resolve_model(None, phase="implement")
        assert result == "model-instance"
        assert built["spec"] == "openai:Qwen-Local-GGUF"
        assert built["cfg"]["name"] == "local-standby"


class TestReadinessGate:
    """A warming server listens before it serves (probe 24/67056e02: the TCP
    check flipped healthy during mini-sglang's warmup and gap_plan died on
    the half-up serve). Transitions TO healthy require a 1-token completion."""

    def _tcp_ok(self, monkeypatch):
        import socket as socket_mod

        class FakeSock:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            socket_mod, "create_connection", lambda addr, timeout: FakeSock()
        )

    def test_recovery_requires_readiness(self, monkeypatch):
        self._tcp_ok(monkeypatch)
        with patch.object(helpers, "_endpoint_ready", return_value=False) as ready:
            assert helpers._endpoint_healthy("http://u:1/v1", "openai:M") is False
        ready.assert_called_once_with("http://u:1/v1", "openai:M")

    def test_recovery_with_ready_backend_is_healthy(self, monkeypatch):
        self._tcp_ok(monkeypatch)
        with patch.object(helpers, "_endpoint_ready", return_value=True):
            assert helpers._endpoint_healthy("http://u:1/v1", "openai:M") is True

    def test_steady_state_healthy_skips_readiness(self, monkeypatch):
        import time

        self._tcp_ok(monkeypatch)
        # Seed known-healthy with an EXPIRED timestamp so the TTL cache
        # misses and the TCP probe re-runs, but the prior state is healthy.
        helpers._ENDPOINT_HEALTH["http://u:1/v1"] = (
            True, time.monotonic() - helpers._HEALTH_TTL_SECONDS - 1,
        )
        with patch.object(helpers, "_endpoint_ready") as ready:
            assert helpers._endpoint_healthy("http://u:1/v1", "openai:M") is True
        ready.assert_not_called()

    def test_no_model_falls_back_to_tcp_only(self, monkeypatch):
        self._tcp_ok(monkeypatch)
        # _endpoint_ready returns True when it has no model to probe with.
        assert helpers._endpoint_healthy("http://u:1/v1") is True


class TestDegradedModeCoRouting:
    """`degrade_with: <peer>` — when the peer's endpoint is down, this
    provider serves from the peer's fallback so a single-stream standby
    holds ONE resident model (batch 1: Qwen fallback + GLM judge swapped
    on Lemonade and requests around each swap dropped, killing gap_plan
    twice)."""

    GLM = {
        "name": "local-glm",
        "model": "openai:GLM-Local",
        "base_url": "http://localhost:8010/v1",
        "degrade_with": "remote-qwen",
    }

    def _lookup(self, monkeypatch):
        _with_lookup(monkeypatch, [self.GLM, PRIMARY, STANDBY])

    def test_peer_down_coroutes_to_peer_fallback(self, monkeypatch):
        self._lookup(monkeypatch)
        health = {
            "http://10.0.0.1:1919/v1": False,   # peer (primary) down
            "http://localhost:8010/v1": True,   # standby healthy
        }
        with patch.object(
            helpers, "_endpoint_healthy",
            side_effect=lambda url, model=None: health[url],
        ):
            cfg, spec = helpers._apply_provider_fallback(dict(self.GLM), self.GLM["model"])
        assert cfg["name"] == "local-standby"
        assert spec == "openai:Qwen-Local-GGUF"

    def test_peer_healthy_keeps_own_model(self, monkeypatch):
        self._lookup(monkeypatch)
        with patch.object(helpers, "_endpoint_healthy", return_value=True):
            cfg, spec = helpers._apply_provider_fallback(dict(self.GLM), self.GLM["model"])
        assert cfg["name"] == "local-glm"
        assert spec == "openai:GLM-Local"

    def test_peer_down_but_no_healthy_standby_keeps_own_model(self, monkeypatch):
        self._lookup(monkeypatch)
        with patch.object(helpers, "_endpoint_healthy", return_value=False):
            cfg, spec = helpers._apply_provider_fallback(dict(self.GLM), self.GLM["model"])
        assert cfg["name"] == "local-glm"

    def test_missing_peer_entry_keeps_own_model(self, monkeypatch):
        _with_lookup(monkeypatch, [self.GLM])  # peer not in providers
        with patch.object(helpers, "_endpoint_healthy", return_value=False):
            cfg, spec = helpers._apply_provider_fallback(dict(self.GLM), self.GLM["model"])
        assert cfg["name"] == "local-glm"
