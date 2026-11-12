import config


def test_api_key_validation_rejects_placeholders():
    assert config.is_configured_api_key(None) is False
    assert config.is_configured_api_key("") is False
    assert config.is_configured_api_key("Your_SILICONFLOW_API_KEY") is False
    assert config.is_configured_api_key("sk-real-value") is True


def test_default_model_selection_order():
    assert config.choose_default_model("sk-deepseek", "sk-glm", True) == "deepseek"
    assert config.choose_default_model(None, "sk-glm", True) == "glm"
    assert config.choose_default_model(None, None, True) == "ollama"
    assert config.choose_default_model(None, None, False) == "deepseek"
