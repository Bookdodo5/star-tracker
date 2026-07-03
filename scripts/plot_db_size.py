import matplotlib.pyplot as plt

sizes_mb = {"Catalog DB": 1041040 / 2**20, "TETRA DB": 19545552 / 2**20}

plt.bar(sizes_mb.keys(), sizes_mb.values(), color=["steelblue", "indianred"])
plt.ylabel("Compiled binary size (MiB)")
plt.title("Compiled database size (object code)")
for i, v in enumerate(sizes_mb.values()):
    plt.text(i, v + 0.3, f"{v:.2f} MB", ha="center")
plt.savefig("outputs/db_size_bar.png", dpi=150)
