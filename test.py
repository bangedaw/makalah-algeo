import numpy as np
import matplotlib.pyplot as plt

def visualize_grover_process(n_qubits, target_index):
    """
    Simulasi Grover dengan visualisasi langkah demi langkah dan plot geometris.
    """
    N = 2**n_qubits
    
    # --- 1. Persiapan Matriks (Konsep: Transformasi Linear) ---
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    H_n = H1
    for _ in range(n_qubits - 1):
        H_n = np.kron(H_n, H1)
        
    # Inisialisasi State |0...0>
    state = np.zeros(N, dtype=np.complex128)
    state[0] = 1.0 
    
    # Buat Superposisi Awal |s>
    state = H_n @ state
    s_vec = state.copy().reshape(-1, 1) # Simpan vektor |s> untuk Diffuser
    
    # Oracle Matrix: I - 2|w><w| (Matriks Diagonal)
    Oracle = np.eye(N, dtype=np.complex128)
    Oracle[target_index, target_index] = -1
    
    # Diffuser Matrix: 2|s><s| - I (Refleksi Householder)
    Diffuser = 2 * (s_vec @ s_vec.conj().T) - np.eye(N)
    
    # --- 2. Setup Geometri untuk Plotting ---
    w_vec = np.zeros(N)
    w_vec[target_index] = 1.0
    
    # Vektor basis |s'> (Gram-Schmidt)
    s_prime = s_vec.flatten() - (np.vdot(w_vec, s_vec.flatten()) * w_vec)
    norm_s_prime = np.linalg.norm(s_prime)
    
    # Hindari pembagian dengan nol jika s_prime sangat kecil
    if norm_s_prime > 1e-10:
        s_prime = s_prime / norm_s_prime
    
    # Fungsi koordinat
    def get_coords(vec):
        y = np.abs(np.vdot(w_vec, vec))       # Proyeksi ke Target
        x = np.abs(np.vdot(s_prime, vec))     # Proyeksi ke Non-Target
        return x, y

    # --- 3. Loop Iterasi Grover ---
    optimal_iter = int(np.pi / 4 * np.sqrt(N))
    print(f"Ruang Pencarian N={N}, Target Indeks={target_index}")
    print(f"Iterasi Optimal Teoritis: {optimal_iter}")
    
    # PERBAIKAN 3: Inisialisasi list kosong dengan benar
    amplitudes_history = []
    coords_history = []
    
    # Simpan kondisi awal
    amplitudes_history.append(state.real.copy())
    coords_history.append(get_coords(state))
    
    for i in range(optimal_iter):
        # a. Terapkan Oracle
        state = Oracle @ state
        
        # b. Terapkan Diffuser
        state = Diffuser @ state
        
        # Simpan data
        amplitudes_history.append(state.real.copy())
        coords_history.append(get_coords(state))

    # --- 4. Visualisasi ---
    fig = plt.figure(figsize=(14, 6))
    
    # Plot A: Evolusi Amplitudo
    ax1 = fig.add_subplot(1, 2, 1)
    
    indices = np.arange(N)
    ax1.bar(indices, amplitudes_history[0], color='blue', alpha=0.3, label='Awal (|s>)')
    ax1.bar(indices, amplitudes_history[-1], color='red', alpha=0.7, label=f'Akhir (Iterasi {optimal_iter})')
    
    # Highlight Target
    ax1.bar([target_index], [amplitudes_history[-1][target_index]], color='green')
    
    ax1.set_xlabel('Index State (Basis Komputasi)')
    ax1.set_ylabel('Amplitudo (Bagian Real)')
    ax1.set_title(f'Amplifikasi Amplitudo pada Target {target_index}')
    ax1.set_ylim(-1.1, 1.1)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.legend()
    
    # Plot B: Rotasi Geometris
    ax2 = fig.add_subplot(1, 2, 2)
    
    # PERBAIKAN 5: Unpacking koordinat (x, y) dengan benar
    xs = [c[0] for c in coords_history]
    ys = [c[1] for c in coords_history]
    
    # Plot jejak rotasi
    ax2.plot(xs, ys, 'o--', color='purple', label='Lintasan Vektor')
    
    # Gambar panah
    for i in range(len(xs)):
        # Sedikit adjustment agar panah terlihat rapi
        ax2.quiver(0, 0, xs[i], ys[i], angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.3)
        # Label langkah
        ax2.text(xs[i], ys[i] + 0.02, f'{i}', fontsize=10, fontweight='bold', color='darkblue')

    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Komponen |s'> (Bukan Solusi)")
    ax2.set_ylabel("Komponen |w> (Solusi)")
    ax2.set_title("Rotasi Vektor Keadaan (2D Plane)")
    ax2.grid(True)
    
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle=':')
    ax2.add_artist(circle)
    
    plt.tight_layout()
    plt.show()

# Jalankan simulasi
if __name__ == "__main__":
    visualize_grover_process(11, 90)