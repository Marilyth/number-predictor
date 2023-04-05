from model import Trainer
from canvas import start_drawing
import sys


if __name__ == "__main__":
    model = Trainer()
    if "test" in sys.argv:
        model.load_model()

        def predict(image) -> str:
            return f"{sorted([(i, round(probability, 2)) for i, probability in enumerate(model.predict(image)) if round(probability, 2) > 0], key=lambda x: x[1], reverse=True)}"
        
        start_drawing(predict)
    else:
        model.train()
        model.save_model()
