import hydra
from mdoc.mdoc.scripts.main_core import run_core

def main(file_path: str, question_string: str):
    global answer
    answer = None

    @hydra.main(config_path="../config", config_name="setting", version_base=None)
    def _main(cfg):
        cfg.input.pdf_path = file_path
        cfg.input.question = question_string
        
        global answer
        answer = run_core(cfg)
        print(answer)

    _main()
    
    return answer

if __name__ == "__main__":
    main(file_path="./files/example.pdf", question_string="What are the main points?")
