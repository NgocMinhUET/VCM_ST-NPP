# Evaluate task performance with real codec integration
from utils.common_utils import load_model_and_data, evaluate_model

def main():
    model = load_model_and_data()
    evaluate_model(model)

if __name__ == "__main__":
    main()
