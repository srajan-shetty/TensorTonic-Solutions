from collections import Counter
def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """
    token_bytes = {i: bytes([i]) for i in range(256)}
    seqs = [list(s.encode("utf-8")) for s in corpus]
    vocab, merges = [], []
    next_id = 256

    while next_id<vocab_size:
        counts = Counter()
        
        for seq in seqs:
            counts.update(zip(seq, seq[1:]))

        if not counts:
            break

        left,right = max(counts, key= lambda p : (counts[p], token_bytes[p[0]], token_bytes[p[1]],p))

        new_id = next_id
        next_id+=1

        token_bytes[new_id] = token_bytes[left]+token_bytes[right]
        merges.append([left,right,new_id])
        vocab.append([new_id, list(token_bytes[new_id])])


        for k,seq in enumerate(seqs):
            out,i,n = [],0,len(seq)

            while i<n:
                if i+1<n and seq[i]==left and seq[i+1]==right:
                    out.append(new_id)
                    i+=2
                else:
                    out.append(seq[i])
                    i+=1

            seqs[k] = out


    return {"vocab": vocab, "merges": merges}

