def test_overview_empty(client):
    response = client.get("/api/overview")

    assert response.status_code == 200

    data = response.json()
    assert data["total_documents"] == 0
    assert data["total_characters"] == 0
    assert data["total_words"] == 0