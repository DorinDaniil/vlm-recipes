from src import evaluate
from src import model as m

if __name__ == "__main__":
    model, tokenizer = m.load()
    evaluate.judge(model, tokenizer)
    print(evaluate.table())
