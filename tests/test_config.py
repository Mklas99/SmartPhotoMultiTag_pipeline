from src.config import TrainConfig, DEFAULT_CLASSES

def test_train_config():
    cfg = TrainConfig()
    assert cfg.epochs > 0
    assert cfg.seed == 42

def test_classes_present():
    assert "person" in DEFAULT_CLASSES
    assert isinstance(DEFAULT_CLASSES, list)
