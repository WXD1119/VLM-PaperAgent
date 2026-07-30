from paper_agent.storage import check_storage_health


def test_storage_health_reports_unconfigured_services_without_network_calls():
    report = check_storage_health()
    assert report.ready is True
    assert {check.name: check.status for check in report.checks} == {
        "redis": "not_configured",
        "mysql": "not_configured",
        "neo4j": "not_configured",
    }
