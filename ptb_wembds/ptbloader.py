import tarfile
import numpy as np
from string import whitespace

def ptb(path):
    '''
    Reads:
        ./simple-examples/data/ptb.train.txt
        ./simple-examples/data/ptb.test.txt
        ./simple-examples/data/ptb.valid.txt
    simple-examples.tgz (I don't know is it real PTB or not)
    '''
    with tarfile.open(path) as f:
        #print('\n'.join(f.getnames()))
        train_text = str(f.extractfile(
                f.getmember('./simple-examples/data/ptb.train.txt')
        ).read(), encoding='utf-8')
        test_text = str(f.extractfile(
                f.getmember('./simple-examples/data/ptb.test.txt')
        ).read(), encoding='utf-8')
        valid_text = str(f.extractfile(
                f.getmember('./simple-examples/data/ptb.valid.txt')
        ).read(), encoding='utf-8')
    return train_text, test_text, valid_text

class Vocabulary:
    def __init__(self, voc_text):
        self.ids = 0
        self.v = {}
        for w in voc_text.split(' '):
            if w in whitespace:
                continue
            if w not in self.v:
                self.v[w] = self.ids
                self.ids += 1
    
    @staticmethod
    def _parse(text):
        s = text.split('\n')
        sw = []
        for w in s:
            cs = w.split(' ')
            if cs:
                del cs[0]
            if cs:
                del cs[-1]
            else:
                continue
            if len(cs) < 2:
                continue
            sw.append(cs)
        return sw
    
    def encode(self, w):
        return self.v[w] if w in self.v else self.v['<unk>']
    
    def decode(self, c):
        if hasattr(self, 'iv'):
            return self.iv[c]
        else:
            self.iv = {v: k for k, v in self.v.items()}
            return self.decode(c)
    
    def skipgram_trainset(self, train_text, window, batch_size):
        s = Vocabulary._parse(train_text)
        while True:
            batch_x, batch_y = [], []
            for sidx in np.random.choice(len(s), batch_size):
                cs = s[sidx]
                wpos = np.random.choice(len(cs))
                rl = len(cs) - wpos
                crp = wpos + window if rl > window else wpos + rl - 1
                clp = wpos - window if wpos > window else 0
                if len(cs) == 2:
                    crange = [p for p in range(clp, crp+1) if p != wpos]
                else:
                    crange = [p for p in range(clp, crp) if p != wpos]
                cpos = np.random.choice(crange)
                #if len(cs) == 2:
                #    print(cs, wpos, clp, crp, crange)
                
                #try:
                #    cpos = np.random.choice(crange)
                #except:
                #    print(cs, wpos, crp, clp, crange)
                #    break
                
                #print(cs)
                #print(self.encode(cs[wpos]), self.encode(cs[cpos]))
                #print(
                #        self.decode(self.encode(cs[wpos])),
                #        self.decode(self.encode(cs[cpos]))
                #)
                #print(wpos, crp, clp, crange, cpos)
                #print(cs[wpos], cs[crp], cs[clp], cs[cpos])
                '''
                batch.append(
                        [
                                self.encode(cs[wpos]),
                                self.encode(cs[cpos])
                        ]
                )
                '''
                batch_x.append(self.encode(cs[wpos]))
                batch_y.append(self.encode(cs[cpos]))
            yield np.array(batch_x), np.array(batch_y).reshape([-1, 1])
            
            
        

if __name__ == '__main__':
    train_text, test_text, valid_text = ptb(
            '/home/ahab/dataset/simple-examples.tgz'
    )
    v = Vocabulary(train_text)
    s = v.skipgram_trainset(train_text, 5, 2)
    for _ in range(3):
        next(s)

