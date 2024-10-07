"Code that simulates error correction in the surface code using stim from scracth"

import numpy as np
import stim
# from noise import NoiseModel
import matplotlib.pyplot as plt
import pymatching
import sinter
from typing import List


def measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index):
    # Measure all z stabilizers

    circuit = stim.Circuit()

    # print(list_z_ancillas_index)

    list_pairs = [[1, -1], [-1, -1], [-1, +1], [+1, +1]]

    # order to measure z stabilizers
    # south-east, #south-west, north-west, north-east

    for xi, yi in list_pairs:

        for ancilla_qubit_idx in list_z_ancillas_index:
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

            if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                # print(ancilla_qubit_idx,data_qubit_idx)
                circuit.append("CNOT", [data_qubit_idx, ancilla_qubit_idx])

            else:
                continue

        circuit.append("TICK")

    return circuit


def measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index):
    # Measure all X stabilizers

    circuit = stim.Circuit()

    circuit.append("H", list_x_ancillas_index)
    circuit.append("TICK")

    list_pairs = [[1, -1], [+1, +1], [-1, +1], [-1, -1]]

    # order to measure z stabilizers
    # south-east, north-east, north-west, #south-west

    for xi, yi in list_pairs:

        for ancilla_qubit_idx in list_x_ancillas_index:
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

            if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                # print(ancilla_qubit_idx,data_qubit_idx)
                circuit.append("CNOT", [ancilla_qubit_idx, data_qubit_idx])

            else:
                continue

        circuit.append("TICK")

    circuit.append("H", list_x_ancillas_index)
    circuit.append("TICK")

    return circuit


def count_logical_errors(circuit: stim.Circuit, num_shots: int):
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    # detector_error_model = circuit.detector_error_model(decompose_errors=False)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors


def surface_code_capacity_zero(Lx, Ly, p):
    "Logical zero code capacity"
    circuit = stim.Circuit()

    Lx_ancilla = 2 * Lx + 1
    Ly_ancilla = 2 * Ly + 1
    # coord_to_index = np.zeros((Lx_ancilla,Ly_ancilla),dtype=int)
    coord_to_index = {}
    index_to_coordinate = []

    # Data qubit coordinates
    qubit_idx = 0
    for yi in range(Ly):
        y = 2 * yi + 1
        for xi in range(Lx):
            x = 2 * xi + 1
            circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, y) + " {}".format(qubit_idx))
            # coord_to_index[x,y] = qubit_idx
            coord_to_index.update({"({},{})".format(x, y): qubit_idx})
            index_to_coordinate.append([x, y])

            qubit_idx += 1

    # Ancilla qubit coordinates

    list_z_ancillas_index = []
    list_x_ancillas_index = []

    for x in range(4, Lx_ancilla, 4):
        circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, 0) + " {}".format(qubit_idx))
        # coord_to_index[x,y] = qubit_idx
        coord_to_index.update({"({},{})".format(x, 0): qubit_idx})
        index_to_coordinate.append([x, 0])

        list_z_ancillas_index.append(qubit_idx)

        qubit_idx += 1

    for y in range(2, Ly_ancilla - 1, 2):
        yi = 2 - y % 4
        # print(yi)
        xs = np.arange(yi, 2 * Lx + yi // 2, 2)
        for dummy, x in enumerate(xs):
            circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, y) + " {}".format(qubit_idx))
            # coord_to_index[x,y] = qubit_idx
            coord_to_index.update({"({},{})".format(x, y): qubit_idx})
            index_to_coordinate.append([x, y])

            if dummy % 2 == 1:
                list_z_ancillas_index.append(qubit_idx)
            elif dummy % 2 == 0:
                list_x_ancillas_index.append(qubit_idx)

            qubit_idx += 1

    for x in range(2, Lx_ancilla - 1, 4):
        circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, Ly_ancilla - 1) + " {}".format(qubit_idx))
        # coord_to_index[x,y] = qubit_idx
        coord_to_index.update({"({},{})".format(x, Ly_ancilla - 1): qubit_idx})
        index_to_coordinate.append([x, Ly_ancilla - 1])
        list_z_ancillas_index.append(qubit_idx)

        qubit_idx += 1

    # print(coord_to_index)
    # print(index_to_coordinate)

    measure_z = measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
    measure_x = measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index)

    circuit.append("R", range(2 * Lx * Ly - 1))
    circuit.append("TICK")

    circuit += measure_x

    # Measure all X stabilizers

    circuit.append("MR", list_x_ancillas_index)
    circuit.append("TICK")

    # round of errors

    data_qubits = ""
    for i in range(Lx * Ly): data_qubits += " {}".format(i)

    # circuit.append_from_stim_program_text("DEPOLARIZE1({})".format(p) + data_qubits)
    circuit.append_from_stim_program_text("X_ERROR({})".format(p) + data_qubits)
    circuit.append("TICK")

    # Error correction begins

    circuit += measure_x
    circuit.append("M", list_x_ancillas_index)

    for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
        offset = (Lx ** 2 - 1) // 2
        coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
        circuit.append_from_stim_program_text(
            "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx, 1 + idx + offset))

    circuit += measure_z
    circuit.append("M", list_z_ancillas_index)

    for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
        coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
        circuit.append_from_stim_program_text("DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}]".format(1 + idx))

    # circuit += measure_x
    # circuit.append("M", list_x_ancillas_index)

    # Measure all data qubits

    circuit.append("M", range(Lx * Ly))

    obs = ""

    for idx in range(1, Lx * Ly + 1): obs += " rec[-{}]".format(idx)

    circuit.append_from_stim_program_text("OBSERVABLE_INCLUDE(0)" + obs)

    return circuit


def surface_code_circuit(distance, noise):
    circuit = stim.Circuit()
    # initialize qubits in |0> state
    # data qubits
    for n in np.arange(2 * distance ** 2):
        circuit.append("R", [n])
    # stabilizer qubits
    for i in np.arange(2 * distance ** 2 - 2):
        circuit.append("R", [2 * distance ** 2 + i])
    circuit.append("TICK")
    # Encoding
    # Measure all stabilizers to project into eigenstate of stabilizers, use stim's detector annotation
    # Z stabilizers
    for i in np.arange(distance):
        for j in np.arange(distance):
            # last stabilizer -> not applied due to constraint
            if i * distance + j == distance ** 2 - 1:
                break
            # horizontal CNOTs
            circuit.append("CX", [i * distance + j, 2 * distance ** 2 + i * distance + j])
            if j % distance == distance - 1:
                circuit.append("CX",
                               [(i - 1) * distance + j + 1, 2 * distance ** 2 + i * distance + j])
            else:
                circuit.append("CX", [i * distance + j + 1, 2 * distance ** 2 + i * distance + j])
                # vertical CNOTs
            circuit.append("CX", [distance ** 2 + i * distance + j,
                                  2 * distance ** 2 + i * distance + j])
            if i % distance == 0:
                circuit.append("CX", [distance ** 2 + (i - 1) * distance + j + distance ** 2,
                                      2 * distance ** 2 + i * distance + j])
            else:
                circuit.append("CX", [distance ** 2 + (i - 1) * distance + j,
                                      2 * distance ** 2 + i * distance + j])
    # X stabilizers
    # Hadamard gates
    for i in np.arange(2 * distance ** 2 + distance ** 2 - 1,
                       2 * distance ** 2 + 2 * distance ** 2 - 2):
        circuit.append("H", [i])
    for i in np.arange(distance):
        for j in np.arange(distance):
            # horizontal CNOTs
            if i * distance + j == distance ** 2 - 1:
                break
            # horizontal CNOTs
            circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                  distance ** 2 + i * distance + j])
            if j % distance == 0:
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      distance ** 2 + (i + 1) * distance + j - 1])
            else:
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      distance ** 2 + i * distance + j - 1])
                # vertical CNOTs
            circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                  i * distance + j])
            if i % distance == distance - 1:
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      (i + 1) * distance + j - distance ** 2])
            else:
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      (i + 1) * distance + j])
        # Hadamard gates
    for i in np.arange(2 * distance ** 2 + distance ** 2 - 1,
                       2 * distance ** 2 + 2 * distance ** 2 - 2):
        circuit.append("H", [i])
        # Measurement of syndrome qubits
    for i in np.arange(2 * distance ** 2 - 2):
        circuit.append("MR", [2 * distance ** 2 + i])
    circuit.append("TICK")
    # Noise
    for i in np.arange(2 * distance ** 2):
        circuit.append("X_ERROR", [i], noise)
    circuit.append("TICK")
    # Measure all stabilizers again:
    # Z stabilizers
    for i in np.arange(distance):
        for j in np.arange(distance):
            # last stabilizer -> not applied due to constraint
            if i * distance + j == distance ** 2 - 1:
                break
                # horizontal CNOTs
            circuit.append("CX", [i * distance + j, 2 * distance ** 2 + i * distance + j])
            if j % distance == distance - 1:
                circuit.append("CX",
                               [(i - 1) * distance + j + 1, 2 * distance ** 2 + i * distance + j])
            else:
                circuit.append("CX", [i * distance + j + 1, 2 * distance ** 2 + i * distance + j])
                # vertical CNOTs
            circuit.append("CX", [distance ** 2 + i * distance + j,
                                  2 * distance ** 2 + i * distance + j])
            if i % distance == 0:
                circuit.append("CX", [distance ** 2 + (i - 1) * distance + j + distance ** 2,
                                      2 * distance ** 2 + i * distance + j])
            else:
                circuit.append("CX", [distance ** 2 + (i - 1) * distance + j,
                                      2 * distance ** 2 + i * distance + j])
        # X stabilizers
        # Hadamard gates
    for i in np.arange(2 * distance ** 2 + distance ** 2 - 1,
                       2 * distance ** 2 + 2 * distance ** 2 - 2):
        circuit.append("H", [i])
    for i in np.arange(distance):
        for j in np.arange(distance):
            # horizontal CNOTs
            if i * distance + j == distance ** 2 - 1:
                break
                # horizontal CNOTs
            circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                  distance ** 2 + i * distance + j])
            if j % distance == 0:
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      distance ** 2 + (i + 1) * distance + j - 1])
            else:
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      distance ** 2 + i * distance + j - 1])
                # vertical CNOTs
            circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                  i * distance + j])
            if i % distance == distance - 1:
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      (i + 1) * distance + j - distance ** 2])
            else:
                circuit.append("CX", [2 * distance ** 2 + distance ** 2 - 1 + i * distance + j,
                                      (i + 1) * distance + j])
        # Hadamard gates
    for i in np.arange(2 * distance ** 2 + distance ** 2 - 1,
                       2 * distance ** 2 + 2 * distance ** 2 - 2):
        circuit.append("H", [i])
        # Measurement of syndrome qubits
    for i in np.arange(2 * distance ** 2 - 2):
        circuit.append("MR", [2 * distance ** 2 + i])
        # Add detectors
    for i in np.arange(distance ** 2 - 1):  # changed something here
        circuit.append_from_stim_program_text("DETECTOR({},{})".format(i // distance, i % distance) + " rec[{}] rec[{}]".format(-2 * distance ** 2 + 2 - 2 * distance ** 2 + 2 + i, -2 * distance ** 2 + 2 + i))
    for i in np.arange(2 * distance ** 2):
        circuit.append("M", [i])
    obs = ""
    for idx in range(1, 2 * distance ** 2 + 1):
        obs += " rec[-{}]".format(idx)
    circuit.append_from_stim_program_text("OBSERVABLE_INCLUDE(0)" + obs)
    return circuit


# p=0.1
# circuit = surface_code_capacity_zero(3,3,p)

# circuit.without_noise().diagram("timeslice-svg")

# print(circuit)
# p=1e-1
# n_shots = 5
fig, ax = plt.subplots(1, 1, constrained_layout=True)

s = surface_code_capacity_zero(3, 3, 0.)
sampler = s.compile_detector_sampler()
detection_events, observable_flips = sampler.sample(3, separate_observables=True)

exit(-1)

Ls = [3, 5, 7]
for L in Ls:
    print("L = {}".format(L))
    num_shots = 200000
    # ps = np.logspace(-3,-1,15)
    #ps = np.linspace(0.001, 0.2, 20)
    # ps = np.array([0])

    ps = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.1, 2, 0.1))))

    ys = []

    for p in ps:
        # noiseModel = NoiseModel.Code_Capacity(p=p)
        #noisy_circuit = surface_code_capacity_zero(L, L, p)
        # noiseModel = NoiseModel.CircuitLevel(p=p)
        # circuit= surface_code_circuit_level(L,L)
        # noisy_circuit = circuit= surface_code_circuit_level(L,L,p)
        # print(circuit)
        # print(circuit)
        # noisy_circuit = noiseModel.noisy_circuit(circuit)
        noisy_circuit = surface_code_circuit(distance=L, noise=p)

        num_errors_sampled = count_logical_errors(noisy_circuit, num_shots)

        ys.append(num_errors_sampled / num_shots)

        # print(noisy_circuit)

    ys = np.array(ys)
    std_err = (ys * (1 - ys) / num_shots) ** 0.5
    # plt.plot(xs, ys,"-x", label="d=" + str(d))
    ax.errorbar(ps, ys, std_err, fmt="-x", label="d=" + str(L))

    # print(ys)

ax.axvline(x=0.109)
ax.legend()
ax.grid()
# ax.set_yscale("log")
ax.set_xlabel("$p$")
ax.set_ylabel("$p_L$")
plt.suptitle("Memory")
# ax.loglog()
# fig.savefig('plot.png')
plt.show()
