from abc import ABC, abstractmethod
import stim
import numpy as np
import src.NN.utils.functions as functions


class QECCode(ABC):
    def __init__(self, distance, noise):
        self.distance = distance
        self.noise = noise
        if distance % 2 == 0:
            raise ValueError("Not optimal distance.")
        self.circuit = self.create_code_instance()

    def circuit_to_png(self):
        diagram = self.circuit.diagram('timeline-svg')
        with open('diagram.svg', 'a') as f:
            f.write(diagram.__str__())

    @abstractmethod
    def create_code_instance(self):
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def get_syndromes(self, n):
        raise NotImplementedError("Subclasses should implement this!")


class RotatedSurfaceCode(QECCode, ABC):
    def __init__(self, distance, noise):
        super().__init__(distance, noise)


class BitFlipSurface(RotatedSurfaceCode):
    def __init__(self, distance, noise):
        super().__init__(distance, noise)
        self.syndromes = functions.number_syndromes(self.distance)

    def create_code_instance(self):
        circuit = stim.Circuit()
        # initialize qubits in |0> state
        # data qubits
        for n in np.arange(self.distance ** 2):
            circuit.append("R", [n])
        # stabilizer qubits
        for i in np.arange(self.distance ** 2 - 1):
            circuit.append("R", [self.distance ** 2 + i])
        # Encoding
        # Hadamard gates at corners
        for n in np.arange(0, self.distance, 2):
            for m in np.arange(0, self.distance, 2):
                circuit.append("H", [self.distance * n + m])
        # CNOTs
        # first all connections pointing down
        single = True
        for i in np.arange(0, self.distance - 1, 2):
            for j in np.arange(0, self.distance, 2):
                if single:
                    circuit.append("CX", [i * self.distance + j, (i + 1) * self.distance + j])
                    single = not single
                else:
                    # down
                    circuit.append("CX", [i * self.distance + j, (i + 1) * self.distance + j])
                    if j > 0:  # qubit not on the left boundary
                        circuit.append("CX", [i * self.distance + j, (i + 1) * self.distance + j - 1])
                    if j < self.distance - 1:  # qubit not on the right boundary
                        circuit.append("CX", [i * self.distance + j, i * self.distance + j + 1])  # to the right
                        circuit.append("CX", [i * self.distance + j, (i + 1) * self.distance + j + 1])
                    single = not single
            if ((self.distance - 1) / 2) % 2 != 0:
                single = not single  # add flip for every second distance
        # now all connections pointing up
        single = False
        for i in np.arange(2, self.distance, 2):
            for j in np.arange(0, self.distance, 2):
                if single:
                    circuit.append("CX", [i * self.distance + j, (i - 1) * self.distance + j])
                    single = not single
                else:
                    # down
                    circuit.append("CX", [i * self.distance + j, (i - 1) * self.distance + j])
                    if j > 0:  # qubit not on the left boundary
                        circuit.append("CX", [i * self.distance + j, (i - 1) * self.distance + j - 1])
                    if j < self.distance - 1:  # qubit not on the right boundary
                        circuit.append("CX", [i * self.distance + j, (i - 1) * self.distance + j + 1])
                    single = not single
            if ((self.distance - 1) / 2) % 2 != 0:
                single = not single  # add flip for every second distance
        # and finally horizontal
        single = True
        for i in np.arange(0, self.distance, 2):
            for j in np.arange(0, self.distance, 2):
                if not single:
                    if j > 0:  # qubit not on the left boundary
                        circuit.append("CX", [i * self.distance + j, i * self.distance + j - 1])  # to the left
                    if j < self.distance - 1:  # qubit not on the right boundary
                        circuit.append("CX", [i * self.distance + j, i * self.distance + j + 1])  # to the right
                    single = not single
                else:
                    single = not single
            if ((self.distance - 1) / 2) % 2 != 0:
                single = not single  # add flip for every second distance

        # Noise
        for i in np.arange(self.distance ** 2):
            circuit.append("X_ERROR", [i], self.noise)
        # Measure stabilizers
        # Z stabilizers
        # connection up
        shift = False
        for i in np.arange(self.distance):
            for j in np.arange((self.distance - 1) / 2):
                bias = 0 if shift else 1
                circuit.append("CX",
                               [int(self.distance * i + 2 * j + bias),
                                int(i * (self.distance - 1) / 2 + j + self.distance ** 2)])
                circuit.append("CX",
                               [int(self.distance * i + 2 * j + bias + 1),
                                int(i * (self.distance - 1) / 2 + j + self.distance ** 2)])
            shift = not shift
        # connection down
        shift = True
        for i in np.arange(self.distance):
            for j in np.arange((self.distance - 1) / 2):
                bias = 0 if shift else 1
                circuit.append("CX", [int(self.distance * i + 2 * j + bias),
                                      int(i * (self.distance - 1) / 2 + j + (
                                              self.distance - 1) / 2 + self.distance ** 2)])
                circuit.append("CX", [int(self.distance * i + 2 * j + bias + 1),
                                      int(i * (self.distance - 1) / 2 + j + (
                                              self.distance - 1) / 2 + self.distance ** 2)])
            shift = not shift
            # measure ancilla qubits
        for i in np.arange((self.distance ** 2 - 1) / 2):
            circuit.append("MR", [int(self.distance ** 2 + i)])
        return circuit

    def get_syndromes(self, n):
        sampler = self.circuit.compile_sampler()
        syndromes = sampler.sample(shots=n)  # shape n x syndrom
        return syndromes


class DepolarizingSurface(RotatedSurfaceCode):

    def __init__(self, distance, noise):
        super().__init__(distance, noise)
        self.syndromes = 2 * functions.number_syndromes(self.distance)

    def create_code_instance(self):
        circuit = stim.Circuit()
        # initialize qubits in |0> state
        # data qubits
        for n in np.arange(self.distance ** 2):
            circuit.append("R", [n])
        # stabilizer qubits
        for i in np.arange(self.distance ** 2 - 1):
            circuit.append("R", [self.distance ** 2 + i])
        circuit.append("TICK")
        # Encoding
        # Hadamard gates at corners
        for n in np.arange(0, self.distance, 2):
            for m in np.arange(0, self.distance, 2):
                circuit.append("H", [self.distance * n + m])
        # CNOTs
        # first all connections pointing down
        single = True
        for i in np.arange(0, self.distance - 1, 2):
            for j in np.arange(0, self.distance, 2):
                if single:
                    circuit.append("CX", [i * self.distance + j, (i + 1) * self.distance + j])
                    single = not single
                else:
                    # down
                    circuit.append("CX", [i * self.distance + j, (i + 1) * self.distance + j])
                    if j > 0:  # qubit not on the left boundary
                        circuit.append("CX", [i * self.distance + j, (i + 1) * self.distance + j - 1])
                    if j < self.distance - 1:  # qubit not on the right boundary
                        circuit.append("CX", [i * self.distance + j, i * self.distance + j + 1])  # to the right
                        circuit.append("CX", [i * self.distance + j, (i + 1) * self.distance + j + 1])
                    single = not single
            if ((self.distance - 1) / 2) % 2 != 0:
                single = not single  # add flip for every second distance
        # now all connections pointing up
        single = False
        for i in np.arange(2, self.distance, 2):
            for j in np.arange(0, self.distance, 2):
                if single:
                    circuit.append("CX", [i * self.distance + j, (i - 1) * self.distance + j])
                    single = not single
                else:
                    # down
                    circuit.append("CX", [i * self.distance + j, (i - 1) * self.distance + j])
                    if j > 0:  # qubit not on the left boundary
                        circuit.append("CX", [i * self.distance + j, (i - 1) * self.distance + j - 1])
                    if j < self.distance - 1:  # qubit not on the right boundary
                        circuit.append("CX", [i * self.distance + j, (i - 1) * self.distance + j + 1])
                    single = not single
            if ((self.distance - 1) / 2) % 2 != 0:
                single = not single  # add flip for every second distance
        # and finally horizontal
        single = True
        for i in np.arange(0, self.distance, 2):
            for j in np.arange(0, self.distance, 2):
                if not single:
                    if j > 0:  # qubit not on the left boundary
                        circuit.append("CX", [i * self.distance + j, i * self.distance + j - 1])  # to the left
                    if j < self.distance - 1:  # qubit not on the right boundary
                        circuit.append("CX", [i * self.distance + j, i * self.distance + j + 1])  # to the right
                    single = not single
                else:
                    single = not single
            if ((self.distance - 1) / 2) % 2 != 0:
                single = not single  # add flip for every second distance
        circuit.append("TICK")
        # Noise
        for i in np.arange(self.distance ** 2):
            circuit.append("X_ERROR", [i], self.noise)
        circuit.append("TICK")
        # Measure stabilizers
        # Z stabilizers
        # connection up
        shift = False
        for i in np.arange(self.distance):
            for j in np.arange((self.distance - 1) / 2):
                bias = 0 if shift else 1
                circuit.append("CX",
                               [int(self.distance * i + 2 * j + bias),
                                int(i * (self.distance - 1) / 2 + j + self.distance ** 2)])
                circuit.append("CX",
                               [int(self.distance * i + 2 * j + bias + 1),
                                int(i * (self.distance - 1) / 2 + j + self.distance ** 2)])
            shift = not shift
        # connection down
        shift = True
        for i in np.arange(self.distance):
            for j in np.arange((self.distance - 1) / 2):
                bias = 0 if shift else 1
                circuit.append("CX", [int(self.distance * i + 2 * j + bias),
                                      int(i * (self.distance - 1) / 2 + j + (
                                              self.distance - 1) / 2 + self.distance ** 2)])
                circuit.append("CX", [int(self.distance * i + 2 * j + bias + 1),
                                      int(i * (self.distance - 1) / 2 + j + (
                                              self.distance - 1) / 2 + self.distance ** 2)])
            shift = not shift
        circuit.append("TICK")
        # X stabilizers
        # Hadamard gates
        for i in np.arange(self.distance ** 2 + int((self.distance ** 2 - 1) / 2), 2 * self.distance ** 2 - 1):
            circuit.append("H", [i])
        # connection to the right
        shift = True
        for i in np.arange(self.distance):
            for j in np.arange((self.distance - 1) / 2):
                bias = 0 if shift else 1
                circuit.append("CX",
                               [int(i * (self.distance - 1) / 2 + j + self.distance ** 2 + (
                                       self.distance ** 2 - 1) / 2),
                                int(i + 2 * j * self.distance + bias * self.distance)])
                circuit.append("CX",
                               [int(i * (self.distance - 1) / 2 + j + self.distance ** 2 + (
                                       self.distance ** 2 - 1) / 2),
                                int(i + 2 * j * self.distance + bias * self.distance + self.distance)])
            shift = not shift
        # connection right
        shift = False
        for i in np.arange(self.distance):
            for j in np.arange((self.distance - 1) / 2):
                bias = 0 if shift else 1
                circuit.append("CX",
                               [int(i * (self.distance - 1) / 2 + j + self.distance ** 2 + (
                                       self.distance ** 2 - 1) / 2 + (self.distance - 1) / 2),
                                int(i + 2 * j * self.distance + bias * self.distance)])
                circuit.append("CX",
                               [int(i * (self.distance - 1) / 2 + j + self.distance ** 2 + (
                                       self.distance ** 2 - 1) / 2 + (self.distance - 1) / 2),
                                int(i + 2 * j * self.distance + bias * self.distance + self.distance)])
            shift = not shift
        # Hadamard gates
        for i in np.arange(self.distance ** 2 + int((self.distance ** 2 - 1) / 2), 2 * self.distance ** 2 - 1):
            circuit.append("H", [i])
        circuit.append("TICK")
        # measure ancilla qubits
        for i in np.arange(self.distance ** 2 - 1):
            circuit.append("MR", [int(self.distance ** 2 + i)])
        return circuit

    def get_syndromes(self, n):
        sampler = self.circuit.compile_sampler()
        syndromes = sampler.sample(shots=n)  # shape n x syndrom
        return syndromes
