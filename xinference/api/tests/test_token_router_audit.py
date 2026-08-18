from xinference.api.oauth2.advanced.audit import classify_endpoint, should_skip_audit


def test_internal_token_router_endpoints_skip_audit():
    assert should_skip_audit("/v1/internal/token-router/instances/register") is True
    assert (
        should_skip_audit("/v1/internal/token-router/instances/router-1/config-ack")
        is True
    )


def test_token_router_management_endpoints_are_admin_audited():
    endpoint = "/v1/token_routers/router-1/enable"
    assert should_skip_audit(endpoint) is False
    assert classify_endpoint(endpoint) == "admin"
