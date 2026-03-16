from main import _aggregate

def test_weight_sum():
    # Test code weights
    w, s, e = 0.5, 0.5, 0.5
    # The absolute values don't matter as much as the internal weighting coefficients
    # In main.py:
    # Code: 0.2 + 0.6 + 0.2 = 1.0
    # Text: 0.3 + 0.2 + 0.5 = 1.0
    
    # We test it by passing 1.0, 1.0, 1.0 and ensuring we get 1.0 back
    code_res = _aggregate(1.0, 1.0, 1.0, "py")
    text_res = _aggregate(1.0, 1.0, 1.0, "txt")
    
    print(f"Code weight sum: {code_res}")
    print(f"Text weight sum: {text_res}")
    
    assert abs(code_res - 1.0) < 1e-9, f"Code weights sum to {code_res}, expected 1.0"
    assert abs(text_res - 1.0) < 1e-9, f"Text weights sum to {text_res}, expected 1.0"
    print("Verification Passed: All algorithm weights sum to 1.0")

if __name__ == "__main__":
    test_weight_sum()
