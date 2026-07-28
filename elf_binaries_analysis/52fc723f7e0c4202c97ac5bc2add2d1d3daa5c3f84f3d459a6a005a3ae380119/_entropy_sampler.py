import sys, os, time, math, glob

def shannon_entropy(data: bytes) -> float:
    if not data: return 0.0
    freq = [0] * 256
    for b in data: freq[b] += 1
    n = len(data)
    ent = 0.0
    for c in freq:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return ent

watch_dir, out_csv, interval = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(out_csv, "w") as f:
    f.write("epoch,mean_entropy,file_count,total_bytes\n")
    while True:
        try:
            files = [x for x in glob.glob(os.path.join(watch_dir, "**", "*"), recursive=True)
                     if os.path.isfile(x)]
            ents, total_bytes = [], 0
            for fp in files:
                try:
                    with open(fp, "rb") as fh:
                        data = fh.read(65536)
                    ents.append(shannon_entropy(data))
                    total_bytes += os.path.getsize(fp)
                except OSError:
                    continue
            mean_e = sum(ents) / len(ents) if ents else 0.0
            f.write(f"{time.time():.3f},{mean_e:.4f},{len(files)},{total_bytes}\n")
            f.flush()
        except KeyboardInterrupt:
            break
        time.sleep(interval)
