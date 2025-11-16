"""MiniMind 对话脚本"""
import torch
from minimind import MiniMindConfig, MiniMindForCausalLM
from tokenizer import SimpleCharTokenizer


class Chat:
    def __init__(self, model_path="./output/minimind_model.pt", 
                 tokenizer_path="./output/tokenizer.pkl", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = SimpleCharTokenizer.load(tokenizer_path)
        self.config = MiniMindConfig(vocab_size=self.tokenizer.get_vocab_size())
        self.model = MiniMindForCausalLM(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.model.to(self.device).eval()
        print(f"✅ 模型已加载到 {self.device}")
    
    def generate(self, prompt, max_tokens=128, temp=0.8):
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, max_new_tokens=max_tokens, temperature=temp,
                do_sample=True, pad_token_id=0, eos_token_id=2
            )
        return self.tokenizer.decode(output_ids[0].tolist())
    
    def chat(self):
        print("\n🤖 开始对话 (输入 'exit' 退出)\n")
        while True:
            try:
                prompt = input("👤 你: ").strip()
                if not prompt or prompt.lower() == 'exit':
                    break
                print("🤖 助手:", self.generate(prompt)[len(prompt):].strip(), "\n")
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    Chat().chat()
