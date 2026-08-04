import torch

class KoJamoTokenizer:
    def __init__(self):
        # 0은 <PAD> 또는 <EMPTY>
        
        self.cho_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.jung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        self.jong_list = [''] + ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        
        # 단독 자음/모음 처리
        self.standalone_cho = {c: i+1 for i, c in enumerate(self.cho_list)}
        self.standalone_jung = {c: i+1 for i, c in enumerate(self.jung_list)}
        self.standalone_jong = {c: i for i, c in enumerate(self.jong_list) if c != '' and c not in self.cho_list}
        
        # 기호, 영어, 숫자 분리
        self.sym_list = list(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n')
        self.sym_vocab = {char: i+1 for i, char in enumerate(self.sym_list)}
        self.unk_token_id = len(self.sym_list) + 1  # 알 수 없는 문자(한자 등)는 기호 트랙의 UNK로 처리
        
        self.reverse_sym = {i+1: char for i, char in enumerate(self.sym_list)}
        self.reverse_sym[self.unk_token_id] = '?'
        
        self.eng_list = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.eng_vocab = {char: i+1 for i, char in enumerate(self.eng_list)}
        self.reverse_eng = {i+1: char for i, char in enumerate(self.eng_list)}
        
        self.num_list = list('0123456789')
        self.num_vocab = {char: i+1 for i, char in enumerate(self.num_list)}
        self.reverse_num = {i+1: char for i, char in enumerate(self.num_list)}
        
    def get_vocab_sizes(self):
        # 초(20), 중(22), 종(28 = 빈종성 포함 27+1, encode()가 실제로 내는 값은 0~27뿐),
        # 기호(34 + UNK + PAD = 36), 영어(52 + PAD = 53), 숫자(10 + PAD = 11)
        return 20, 22, 28, len(self.sym_list) + 2, len(self.eng_list) + 1, len(self.num_list) + 1
        
    def encode(self, text):
        encoded = []
        for char in text:
            if '가' <= char <= '힣':
                code = ord(char) - 44032
                cho = (code // 588) + 1
                jung = ((code % 588) // 28) + 1
                jong = code % 28
                encoded.append([cho, jung, jong, 0, 0, 0])
            elif char in self.standalone_cho:
                encoded.append([self.standalone_cho[char], 0, 0, 0, 0, 0])
            elif char in self.standalone_jung:
                encoded.append([0, self.standalone_jung[char], 0, 0, 0, 0])
            elif char in self.standalone_jong:
                encoded.append([0, 0, self.standalone_jong[char], 0, 0, 0])
            elif char in self.eng_vocab:
                encoded.append([0, 0, 0, 0, self.eng_vocab[char], 0])
            elif char in self.num_vocab:
                encoded.append([0, 0, 0, 0, 0, self.num_vocab[char]])
            elif char in self.sym_vocab:
                encoded.append([0, 0, 0, self.sym_vocab[char], 0, 0])
            else:
                # 미지원 문자는 기호 트랙의 UNK로 인코딩
                encoded.append([0, 0, 0, self.unk_token_id, 0, 0])
                
        return torch.tensor(encoded, dtype=torch.long)
        
    def decode(self, tensor):
        text = ""
        for step in tensor:
            cho, jung, jong, sym, eng, num = step.tolist()
            
            if sym != 0:
                text += self.reverse_sym.get(sym, '?')
            elif eng != 0:
                text += self.reverse_eng.get(eng, '?')
            elif num != 0:
                text += self.reverse_num.get(num, '?')
            else:
                if cho > 0 and jung > 0: # 완전한 한글
                    char_code = (cho - 1) * 588 + (jung - 1) * 28 + jong
                    text += chr(char_code + 44032)
                elif cho > 0 and jung == 0 and jong == 0: # 단독 초성
                    text += self.cho_list[cho - 1]
                elif cho == 0 and jung > 0 and jong == 0: # 단독 중성
                    text += self.jung_list[jung - 1]
                elif cho == 0 and jung == 0 and jong > 0: # 단독 종성
                    text += self.jong_list[jong]
                elif cho == 0 and jung == 0 and jong == 0:
                    pass # Empty
                else:
                    text += '_'
        return text

if __name__ == "__main__":
    tokenizer = KoJamoTokenizer()
    sample_text = "ㄱㄴㅏ (Hello!) 1234 漢字"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print("=" * 40)
    print(f"원본: {sample_text}")
    print(f"복원: {decoded}")
    print(f"형태: {encoded.shape}")
    print(f"보캡 사이즈: {tokenizer.get_vocab_sizes()}")
    print("=" * 40)
