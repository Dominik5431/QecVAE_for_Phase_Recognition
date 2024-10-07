from abc import ABC, abstractmethod
import stim
import numpy as np
from pathlib import Path
from random import random


class QECCode(ABC):
    def __init__(self, distance, noise, random_flip: bool = False, noise_model: str = ''):
        self.distance = distance
        self.noise = noise
        self.random_flip = random_flip
        if distance % 2 == 0:
            raise ValueError("Not optimal distance.")
        self.circuit = self.create_code_instance()

    def circuit_to_png(self):
        diagram = self.circuit.diagram('timeline-svg')
        with open(str(Path().resolve().parent) + "/data/diagram_pheno.svg", 'w') as f:
            f.write(diagram.__str__())

    @abstractmethod
    def create_code_instance(self):
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def get_syndromes(self, n, flip: bool = False, supervised: bool = False, only_syndromes: bool = False):
        raise NotImplementedError("Subclasses should implement this!")


class BitFlipToricCode(QECCode):
    def __init__(self, distance, noise, random_flip):
        super().__init__(distance, noise, random_flip)
        self.num_syndromes = 2 * distance ** 2 - 2

    def create_code_instance(self):
        circuit = stim.Circuit()
        # initialize qubits in |0> state
        # data qubits
        for n in np.arange(2 * self.distance ** 2):
            circuit.append("R", [n])
        # stabilizer qubits
        for i in np.arange(2 * self.distance ** 2 - 2):
            circuit.append("R", [2 * self.distance ** 2 + i])
        circuit.append("TICK")
        # Encoding
        # Measure all stabilizers to project into eigenstate of stabilizers, use stim's detector annotation
        # Z stabilizers
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # last stabilizer -> not applied due to constraint
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [i * self.distance + j, 2 * self.distance ** 2 + i * self.distance + j])
                if j % self.distance == self.distance - 1:
                    circuit.append("CX",
                                   [(i - 1) * self.distance + j + 1, 2 * self.distance ** 2 + i * self.distance + j])
                else:
                    circuit.append("CX", [i * self.distance + j + 1, 2 * self.distance ** 2 + i * self.distance + j])
                # vertical CNOTs
                circuit.append("CX", [self.distance ** 2 + i * self.distance + j,
                                      2 * self.distance ** 2 + i * self.distance + j])
                if i % self.distance == 0:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j + self.distance ** 2,
                                          2 * self.distance ** 2 + i * self.distance + j])
                else:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j,
                                          2 * self.distance ** 2 + i * self.distance + j])
        # X stabilizers
        # Hadamard gates
        for i in np.arange(2 * self.distance ** 2 + self.distance ** 2 - 1,
                           2 * self.distance ** 2 + 2 * self.distance ** 2 - 2):
            circuit.append("H", [i])
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # horizontal CNOTs
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                      self.distance ** 2 + i * self.distance + j])
                if j % self.distance == 0:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          self.distance ** 2 + (i + 1) * self.distance + j - 1])
                else:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          self.distance ** 2 + i * self.distance + j - 1])
                # vertical CNOTs
                circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                      i * self.distance + j])
                if i % self.distance == self.distance - 1:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          (i + 1) * self.distance + j - self.distance ** 2])
                else:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          (i + 1) * self.distance + j])
        # Hadamard gates
        for i in np.arange(2 * self.distance ** 2 + self.distance ** 2 - 1,
                           2 * self.distance ** 2 + 2 * self.distance ** 2 - 2):
            circuit.append("H", [i])
        # Measurement of syndrome qubits
        for i in np.arange(2 * self.distance ** 2 - 2):
            circuit.append("MR", [2 * self.distance ** 2 + i])
        circuit.append("TICK")
        # Noise
        for i in np.arange(2 * self.distance ** 2):
            circuit.append("X_ERROR", [i], self.noise)
        circuit.append("TICK")
        # Measure all stabilizers again:
        # Z stabilizers
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # last stabilizer -> not applied due to constraint
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [i * self.distance + j, 2 * self.distance ** 2 + i * self.distance + j])
                if j % self.distance == self.distance - 1:
                    circuit.append("CX",
                                   [(i - 1) * self.distance + j + 1, 2 * self.distance ** 2 + i * self.distance + j])
                else:
                    circuit.append("CX", [i * self.distance + j + 1, 2 * self.distance ** 2 + i * self.distance + j])
                # vertical CNOTs
                circuit.append("CX", [self.distance ** 2 + i * self.distance + j,
                                      2 * self.distance ** 2 + i * self.distance + j])
                if i % self.distance == 0:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j + self.distance ** 2,
                                          2 * self.distance ** 2 + i * self.distance + j])
                else:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j,
                                          2 * self.distance ** 2 + i * self.distance + j])
        # X stabilizers
        # Hadamard gates
        for i in np.arange(2 * self.distance ** 2 + self.distance ** 2 - 1,
                           2 * self.distance ** 2 + 2 * self.distance ** 2 - 2):
            circuit.append("H", [i])
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # horizontal CNOTs
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                      self.distance ** 2 + i * self.distance + j])
                if j % self.distance == 0:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          self.distance ** 2 + (i + 1) * self.distance + j - 1])
                else:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          self.distance ** 2 + i * self.distance + j - 1])
                # vertical CNOTs
                circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                      i * self.distance + j])
                if i % self.distance == self.distance - 1:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          (i + 1) * self.distance + j - self.distance ** 2])
                else:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          (i + 1) * self.distance + j])
        # Hadamard gates
        for i in np.arange(2 * self.distance ** 2 + self.distance ** 2 - 1,
                           2 * self.distance ** 2 + 2 * self.distance ** 2 - 2):
            circuit.append("H", [i])
        # Measurement of syndrome qubits
        for i in np.arange(2 * self.distance ** 2 - 2):
            circuit.append("MR", [2 * self.distance ** 2 + i])
        # Add detectors
        for i in np.arange(2 * self.distance ** 2 - 2):
            circuit.append("DETECTOR",
                           [stim.target_rec(-2 * self.distance ** 2 + 2 - 2 * self.distance ** 2 + 2 + i),
                            stim.target_rec(-2 * self.distance ** 2 + 2 + i)])
        return circuit

    def get_syndromes(self, n, flip: bool = False, supervised: bool = False, only_syndromes: bool = False):
        sampler = self.circuit.compile_detector_sampler()
        syndromes = sampler.sample(shots=n)
        # Add here already the last redundant syndrome to have square shape
        z_add = 1
        x_add = 1
        syndromes = np.array(list(map(lambda y: np.where(y, -1, 1), syndromes)))  # tried on 08.05.
        syndromes_temp = []
        syndromes_final = []
        flips = []
        for s in syndromes:
            z_syndromes = s[:int(0.5 * len(s))]
            x_syndromes = s[int(0.5 * len(s)):]
            for z in z_syndromes:
                z_add *= z
            z_syndromes = np.append(z_syndromes, z_add)
            x_add = 1
            for x in x_syndromes:
                x_add *= x
            x_syndromes = np.append(x_syndromes, x_add)
            syndrome_shot = np.append(z_syndromes, x_syndromes)
            syndromes_temp.append(syndrome_shot)
        if self.random_flip:
            if not flip and not supervised:
                for syndrome in syndromes_temp:
                    if random() < 0.5:
                        syndromes_final.append(syndrome)
                        flips.append(1)
                    else:
                        syndromes_final.append(-syndrome)
                        flips.append(-1)
                return syndromes_final, flips
            else:
                syndromes_final = np.array(list(map(lambda y: y if random() < 0.5 else -y, syndromes_temp)))
            # syndromes_final = list(map(lambda y: (y, 1) if random() < 0.5 else (-y, -1), syndromes_final))
            # syndromes_final = np.array(list(map(lambda y: y if random() < 0.5 else -y, syndromes_final)))
        else:
            syndromes_final = syndromes_temp
        return syndromes_final


class DepolarizingToricCode(QECCode):
    def __init__(self, distance, noise, random_flip):
        super().__init__(distance, noise, random_flip)
        self.num_syndromes = 2 * distance ** 2 - 2

    def create_code_instance(self):
        circuit = stim.Circuit()
        n = 2 * self.distance ** 2  # number of physical qubits
        l = 4 * self.distance  # number of logical operators to measure
        s = 2 * self.distance ** 2 - 2  # number of stabilizer qubits
        k = 2  # number of logical qubits and therefore reference qubits
        # initialize qubits in |0> state
        # reference qubits
        for i in np.arange(k):
            circuit.append("R", [i])
        # data qubits
        for i in np.arange(n):
            circuit.append("R", [k + i])
        # stabilizer qubits
        for i in np.arange(s):
            circuit.append("R", [k + n + i])
        # logical operator ancilla qubits for all 4 logical operators (X_L1, X_L2, Z_L1, Z_L2)
        for i in np.arange(l):
            circuit.append("R", [k + n + s + i])
        circuit.append("TICK")
        # Encoding of logical states
        # Measure all stabilizers to project into eigenstate of stabilizers, use stim's detector annotation
        # Z stabilizers
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # last stabilizer -> not applied due to constraint
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [i * self.distance + j + k, 2 * self.distance ** 2 + i * self.distance + j + k])
                if j % self.distance == self.distance - 1:
                    circuit.append("CX",
                                   [(i - 1) * self.distance + j + 1 + k,
                                    2 * self.distance ** 2 + i * self.distance + j + k])
                else:
                    circuit.append("CX",
                                   [i * self.distance + j + 1 + k, 2 * self.distance ** 2 + i * self.distance + j + k])
                # vertical CNOTs
                circuit.append("CX", [self.distance ** 2 + i * self.distance + j + k,
                                      2 * self.distance ** 2 + i * self.distance + j + k])
                if i % self.distance == 0:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j + self.distance ** 2 + k,
                                          2 * self.distance ** 2 + i * self.distance + j + k])
                else:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j + k,
                                          2 * self.distance ** 2 + i * self.distance + j + k])
        # X stabilizers
        # Hadamard gates
        for i in np.arange(int(s / 2)):
            circuit.append("H", [k + n + int(s / 2) + i])
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # horizontal CNOTs
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                      self.distance ** 2 + i * self.distance + j + k])
                if j % self.distance == 0:
                    circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                          self.distance ** 2 + (i + 1) * self.distance + j - 1 + k])
                else:
                    circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                          self.distance ** 2 + i * self.distance + j - 1 + k])
                # vertical CNOTs
                circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                      i * self.distance + j + k])
                if i % self.distance == self.distance - 1:
                    circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                          (i + 1) * self.distance + j - self.distance ** 2 + k])
                else:
                    circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                          (i + 1) * self.distance + j + k])
        # Hadamard gates
        for i in np.arange(int(s / 2)):
            circuit.append("H", [k + n + int(s / 2) + i])
        # Measurement of syndrome qubits
        for i in np.arange(s):
            circuit.append("MR", [k + n + i])
        circuit.append("TICK")
        # Measurement of all logical operators via ancilla qubits / measurement of Bell-state stabilizers between reference and logical qubits
        # Hadamard on ancilla qubits that measure logical X operators
        for i in np.arange(int(l / 2)):
            circuit.append("H", [k + n + s + int(l / 2) + i])
        # Measurement vertical Z_L
        for i in np.arange(self.distance):
            circuit.append("CX", [0, k + n + s + i])
            for j in np.arange(self.distance):
                circuit.append("CX", [k + i + j * self.distance, k + n + s + i])
        # Measurement horizontal Z_L
        for i in np.arange(self.distance):
            circuit.append("CX", [1, k + n + s + self.distance + i])
            for j in np.arange(self.distance):
                circuit.append("CX", [self.distance ** 2 + i * self.distance + j + k, k + n + s + self.distance + i])
        # Measurement vertical X_L
        for i in np.arange(self.distance):
            circuit.append("CX", [k + n + s + 2 * self.distance + i, 1])
            for j in np.arange(self.distance):
                circuit.append("CX",
                               [k + n + s + 2 * self.distance + i, self.distance ** 2 + i + j * self.distance + k])
        # Measurement horizontal X_L
        for i in np.arange(self.distance):
            circuit.append("CX", [k + n + s + 3 * self.distance + i, 0])
            for j in np.arange(self.distance):
                circuit.append("CX", [k + n + s + 3 * self.distance + i, i * self.distance + j + k])
        # Hadamard on ancilla qubits that measure logical X operators
        for i in np.arange(int(l / 2)):
            circuit.append("H", [k + n + s + int(l / 2) + i])

        # Measurement of logical operator ancilla qubits to project on eigenstate
        for i in np.arange(l):
            circuit.append("MR", [k + n + s + i])
        circuit.append("TICK")
        # Noise
        for i in np.arange(n):
            circuit.append("DEPOLARIZE1", [k + i], self.noise)
        circuit.append("TICK")
        # Measure all stabilizers again:
        # Z stabilizers
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # last stabilizer -> not applied due to constraint
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [i * self.distance + j + k, 2 * self.distance ** 2 + i * self.distance + j + k])
                if j % self.distance == self.distance - 1:
                    circuit.append("CX",
                                   [(i - 1) * self.distance + j + 1 + k,
                                    2 * self.distance ** 2 + i * self.distance + j + k])
                else:
                    circuit.append("CX",
                                   [i * self.distance + j + 1 + k, 2 * self.distance ** 2 + i * self.distance + j + k])
                # vertical CNOTs
                circuit.append("CX", [self.distance ** 2 + i * self.distance + j + k,
                                      2 * self.distance ** 2 + i * self.distance + j + k])
                if i % self.distance == 0:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j + self.distance ** 2 + k,
                                          2 * self.distance ** 2 + i * self.distance + j + k])
                else:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j + k,
                                          2 * self.distance ** 2 + i * self.distance + j + k])
        # X stabilizers
        # Hadamard gates
        for i in np.arange(int(s / 2)):
            circuit.append("H", [k + n + int(s / 2) + i])
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # horizontal CNOTs
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                      self.distance ** 2 + i * self.distance + j + k])
                if j % self.distance == 0:
                    circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                          self.distance ** 2 + (i + 1) * self.distance + j - 1 + k])
                else:
                    circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                          self.distance ** 2 + i * self.distance + j - 1 + k])
                # vertical CNOTs
                circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                      i * self.distance + j + k])
                if i % self.distance == self.distance - 1:
                    circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                          (i + 1) * self.distance + j - self.distance ** 2 + k])
                else:
                    circuit.append("CX", [n + int(s / 2) + i * self.distance + j + k,
                                          (i + 1) * self.distance + j + k])
        # Hadamard gates
        for i in np.arange(int(s / 2)):
            circuit.append("H", [k + n + int(s / 2) + i])
        # Measurement of syndrome qubits
        for i in np.arange(s):
            circuit.append("MR", [k + n + i])
        circuit.append("TICK")
        # Measurement of all logical operators via ancilla qubits / measurement of Bell-state stabilizers between reference and logical qubits
        # Hadamard on ancilla qubits that measure logical X operators
        for i in np.arange(int(l / 2)):
            circuit.append("H", [k + n + s + int(l / 2) + i])
        # Measurement vertical Z_L
        for i in np.arange(self.distance):
            circuit.append("CX", [0, k + n + s + i])
            for j in np.arange(self.distance):
                circuit.append("CX", [k + i + j * self.distance, k + n + s + i])
        # Measurement horizontal Z_L
        for i in np.arange(self.distance):
            circuit.append("CX", [1, k + n + s + self.distance + i])
            for j in np.arange(self.distance):
                circuit.append("CX", [self.distance ** 2 + i * self.distance + j + k, k + n + s + self.distance + i])
        # Measurement vertical X_L
        for i in np.arange(self.distance):
            circuit.append("CX", [k + n + s + 2 * self.distance + i, 1])
            for j in np.arange(self.distance):
                circuit.append("CX",
                               [k + n + s + 2 * self.distance + i, self.distance ** 2 + i + j * self.distance + k])
        # Measurement horizontal X_L
        for i in np.arange(self.distance):
            circuit.append("CX", [k + n + s + 3 * self.distance + i, 0])
            for j in np.arange(self.distance):
                circuit.append("CX", [k + n + s + 3 * self.distance + i, i * self.distance + j + k])
        # Hadamard on ancilla qubits that measure logical X operators
        for i in np.arange(int(l / 2)):
            circuit.append("H", [k + n + s + int(l / 2) + i])

        # Measurement of logical operator ancilla qubits to project on eigenstate
        for i in np.arange(l):
            circuit.append("MR", [k + n + s + i])
        circuit.append("TICK")
        # Add detectors
        for i in np.arange(2 * self.distance ** 2 - 2):
            circuit.append("DETECTOR",
                           [stim.target_rec(-2 * s - 2 * l + i),
                            stim.target_rec(-s - l + i)])
        for i in np.arange(4 * self.distance):
            circuit.append("DETECTOR",
                           [stim.target_rec(-s - 2 * l + i),
                            stim.target_rec(-l + i)])
        return circuit

    def get_syndromes(self, n, flip: bool = False, supervised: bool = False, only_syndromes: bool = False):
        def add_last_stabilizer(syndromes):
            syndromes_added = []
            for s in syndromes:
                z_syndromes = s[:int(0.5 * len(s))]
                x_syndromes = s[int(0.5 * len(s)):]
                z_add = 1
                for z in z_syndromes:
                    z_add *= z
                z_syndromes = np.append(z_syndromes, z_add)
                x_add = 1
                for x in x_syndromes:
                    x_add *= x
                x_syndromes = np.append(x_syndromes, x_add)
                syndrome_shot = np.append(z_syndromes, x_syndromes)
                syndromes_added.append(syndrome_shot)
            return syndromes_added

        sampler = self.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=n)
        # Add here already the last redundant syndrom to have square shape
        samples = np.array(list(map(lambda y: np.where(y, -1, 1), samples)))  # tried on 08.05.

        syndromes_before_flip = add_last_stabilizer(samples[:, :2 * self.distance ** 2 - 2])

        syndromes_final = []
        flips = []
        logical = samples[:, 2 * self.distance ** 2 - 2:]
        logical_final = []

        output = ()

        if self.random_flip:
            for i, syndrome in enumerate(syndromes_before_flip):
                if random() < 0.5:
                    syndromes_final.append(syndrome)
                    flips.append(1)
                    logical_final.append(logical[i])
                else:
                    syndromes_final.append(-syndrome)
                    flips.append(-1)
                    logical_final.append(-logical[i])
        else:
            syndromes_final = syndromes_before_flip
            logical_final = logical
        output = output + (syndromes_final,)
        output = output + (logical_final,)
        if not flip:
            output = output + (flips,)
        return output


class ToricCodePheno(QECCode):
    def __init__(self, distance, noise, random_flip):
        super().__init__(distance, noise, random_flip)

    def create_code_instance(self):
        circuit = stim.Circuit()
        # initialize qubits in |0> state
        # data qubits
        for n in np.arange(2 * self.distance ** 2):
            circuit.append("R", [n])
        # stabilizer qubits
        for i in np.arange(2 * self.distance ** 2 - 2):
            circuit.append("R", [2 * self.distance ** 2 + i])
        circuit.append("TICK")
        # Encoding
        # Measure all stabilizers to project into eigenstate of stabilizers, use stim's detector annotation
        # Z stabilizers
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # last stabilizer -> not applied due to constraint
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [i * self.distance + j, 2 * self.distance ** 2 + i * self.distance + j])
                if j % self.distance == self.distance - 1:
                    circuit.append("CX",
                                   [(i - 1) * self.distance + j + 1, 2 * self.distance ** 2 + i * self.distance + j])
                else:
                    circuit.append("CX", [i * self.distance + j + 1, 2 * self.distance ** 2 + i * self.distance + j])
                # vertical CNOTs
                circuit.append("CX", [self.distance ** 2 + i * self.distance + j,
                                      2 * self.distance ** 2 + i * self.distance + j])
                if i % self.distance == 0:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j + self.distance ** 2,
                                          2 * self.distance ** 2 + i * self.distance + j])
                else:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j,
                                          2 * self.distance ** 2 + i * self.distance + j])
        # X stabilizers
        # Hadamard gates
        for i in np.arange(2 * self.distance ** 2 + self.distance ** 2 - 1,
                           2 * self.distance ** 2 + 2 * self.distance ** 2 - 2):
            circuit.append("H", [i])
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # horizontal CNOTs
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                      self.distance ** 2 + i * self.distance + j])
                if j % self.distance == 0:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          self.distance ** 2 + (i + 1) * self.distance + j - 1])
                else:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          self.distance ** 2 + i * self.distance + j - 1])
                # vertical CNOTs
                circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                      i * self.distance + j])
                if i % self.distance == self.distance - 1:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          (i + 1) * self.distance + j - self.distance ** 2])
                else:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          (i + 1) * self.distance + j])
        # Hadamard gates
        for i in np.arange(2 * self.distance ** 2 + self.distance ** 2 - 1,
                           2 * self.distance ** 2 + 2 * self.distance ** 2 - 2):
            circuit.append("H", [i])
        # Measurement of syndrome qubits
        for i in np.arange(2 * self.distance ** 2 - 2):
            circuit.append("MR", [2 * self.distance ** 2 + i])
        circuit.append("TICK")

        # TODO hier repeat, structure REPEAT n { text }
        print("REPEAT 5 {\n" + self.syndrome_measurement() + "\n}", sep='')
        circuit.append_from_stim_program_text("REPEAT 5 {\nX 0 1\n}")
        raise Exception
        circuit.append_from_stim_program_text("REPEAT 5 {" + self.syndrome_measurement() + "}")

        return circuit

    def syndrome_measurement(self) -> str:
        circuit = stim.Circuit()
        # Noise
        for i in np.arange(2 * self.distance ** 2):
            circuit.append("X_ERROR", [i], self.noise)
        circuit.append("TICK")
        # Measure all stabilizers again:
        # Z stabilizers
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # last stabilizer -> not applied due to constraint
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [i * self.distance + j, 2 * self.distance ** 2 + i * self.distance + j])
                if j % self.distance == self.distance - 1:
                    circuit.append("CX",
                                   [(i - 1) * self.distance + j + 1, 2 * self.distance ** 2 + i * self.distance + j])
                else:
                    circuit.append("CX", [i * self.distance + j + 1, 2 * self.distance ** 2 + i * self.distance + j])
                # vertical CNOTs
                circuit.append("CX", [self.distance ** 2 + i * self.distance + j,
                                      2 * self.distance ** 2 + i * self.distance + j])
                if i % self.distance == 0:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j + self.distance ** 2,
                                          2 * self.distance ** 2 + i * self.distance + j])
                else:
                    circuit.append("CX", [self.distance ** 2 + (i - 1) * self.distance + j,
                                          2 * self.distance ** 2 + i * self.distance + j])
        # X stabilizers
        # Hadamard gates
        for i in np.arange(2 * self.distance ** 2 + self.distance ** 2 - 1,
                           2 * self.distance ** 2 + 2 * self.distance ** 2 - 2):
            circuit.append("H", [i])
        for i in np.arange(self.distance):
            for j in np.arange(self.distance):
                # horizontal CNOTs
                if i * self.distance + j == self.distance ** 2 - 1:
                    break
                # horizontal CNOTs
                circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                      self.distance ** 2 + i * self.distance + j])
                if j % self.distance == 0:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          self.distance ** 2 + (i + 1) * self.distance + j - 1])
                else:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          self.distance ** 2 + i * self.distance + j - 1])
                # vertical CNOTs
                circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                      i * self.distance + j])
                if i % self.distance == self.distance - 1:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          (i + 1) * self.distance + j - self.distance ** 2])
                else:
                    circuit.append("CX", [2 * self.distance ** 2 + self.distance ** 2 - 1 + i * self.distance + j,
                                          (i + 1) * self.distance + j])
        # Hadamard gates
        for i in np.arange(2 * self.distance ** 2 + self.distance ** 2 - 1,
                           2 * self.distance ** 2 + 2 * self.distance ** 2 - 2):
            circuit.append("H", [i])
        # Measurement of syndrome qubits
        for i in np.arange(2 * self.distance ** 2 - 2):
            circuit.append("MR", [2 * self.distance ** 2 + i])
        # Add detectors
        for i in np.arange(2 * self.distance ** 2 - 2):
            circuit.append("DETECTOR",
                           [stim.target_rec(-2 * self.distance ** 2 + 2 - 2 * self.distance ** 2 + 2 + i),
                            stim.target_rec(-2 * self.distance ** 2 + 2 + i)])
        return circuit.__str__()

    def get_syndromes(self, n):
        sampler = self.circuit.compile_detector_sampler()
        syndromes = sampler.sample(shots=n)
        # Add here already the last redundant syndrom to have square shape
        syndromes_final = []
        for s in syndromes:
            z_syndromes = s[:int(0.5 * len(s))]
            x_syndromes = s[int(0.5 * len(s)):]
            z_add = 1
            for z in z_syndromes:
                z_add *= z
            z_syndromes = np.append(z_syndromes, z_add)
            x_add = 1
            for x in x_syndromes:
                x_add *= x
            x_syndromes = np.append(x_syndromes, x_add)
            syndrome_shot = np.append(z_syndromes, x_syndromes)
            syndromes_final.append(syndrome_shot)
        if self.random_flip:
            return list(map(lambda y: np.where(y, -1, 1) if random() < 0.5 else np.where(y, 1, -1), syndromes_final))
        else:
            return list(map(lambda y: np.where(y, -1, 1), syndromes_final))  # tried on 08.05.


class SurfaceCode(QECCode):
    def __init__(self, distance, noise, noise_model):
        super().__init__(distance, noise)

    def measure_all_z(self, coord_to_index, index_to_coordinate, list_z_ancillas_index):
        circuit = stim.Circuit()
        list_pairs = [[+1, -1], [-1, -1], [-1, +1], [+1, +1]]

        for xi, yi in list_pairs:
            for ancilla_qubit_idx in list_z_ancillas_index:
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

                if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                    data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                    circuit.append("CNOT", [data_qubit_idx, ancilla_qubit_idx])

        circuit.append("TICK")

        return circuit

    def measure_all_x(self, coord_to_index, index_to_coordinate, list_x_ancillas_index):
        circuit = stim.Circuit()

        circuit.append("H", list_x_ancillas_index)
        circuit.append("TICK")

        list_pairs = [[+1, -1], [-1, -1], [-1, +1], [+1, +1]]

        for xi, yi in list_pairs:
            for ancilla_qubit_idx in list_x_ancillas_index:
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

                if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                    data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                    circuit.append("CNOT", [ancilla_qubit_idx, data_qubit_idx])

        circuit.append("TICK")

        circuit.append("H", list_x_ancillas_index)
        circuit.append("TICK")

        return circuit

    def measure_bell_stabilizers(self, coord_to_index, reference_qubit_index, reference_ancillas_index, Ly, Lx):
        circuit = stim.Circuit()

        # Z_R Z_L stabilizer
        for i in range(Ly):
            circuit.append("CNOT", [reference_qubit_index, reference_ancillas_index[i]])
            for xi in range(Lx):
                x = 2 * xi + 1
                circuit.append("CNOT", [coord_to_index["({},{})".format(x, 2 * i + 1)], reference_ancillas_index[i]])
        circuit.append("TICK")

        # X_R X_L stabilizer
        for i in range(Lx):
            circuit.append("H", reference_ancillas_index[Ly + i])
            circuit.append("CNOT", [reference_ancillas_index[Ly + i], reference_qubit_index])
            for yi in range(Ly):
                y = 2 * yi + 1
                circuit.append("CNOT", [reference_ancillas_index[Ly + i], coord_to_index["({},{})".format(2 * i + 1, y)]])
            circuit.append("H", reference_ancillas_index[Ly + i])
        circuit.append("TICK")

        return circuit

    def create_code_instance(self):
        circuit = stim.Circuit()

        Lx, Ly = self.distance, self.distance
        Lx_ancilla, Ly_ancilla = 2 * Lx + 1, 2 * Ly + 1

        coord_to_index = {}
        index_to_coordinate = []

        # data qubit coordinates
        qubit_idx = 0
        for yi in range(Ly):
            y = 2 * yi + 1
            for xi in range(Lx):
                x = 2 * xi + 1
                circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x,y) + " {}".format(qubit_idx))
                coord_to_index.update({"({},{})".format(x, y): qubit_idx})
                index_to_coordinate.append([x, y])

                qubit_idx += 1

        # ancilla qubit coordinates

        list_z_ancillas_index = []
        list_x_ancillas_index = []
        list_data_index = []

        for i in range(Lx * Ly):
            list_data_index.append(i)

        for x in range(2, Lx_ancilla - 1, 4):
            circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, 0) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(x, 0): qubit_idx})
            index_to_coordinate.append([x, 0])

            list_x_ancillas_index.append(qubit_idx)

            qubit_idx += 1

        for y in range(2, Ly_ancilla - 1, 2):
            yi = y % 4
            xs = range(yi, 2 * Lx + yi // 2, 2)
            for idx, x in enumerate(xs):
                circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, y) + " {}".format(qubit_idx))
                coord_to_index.update({"({},{})".format(x, y): qubit_idx})
                index_to_coordinate.append([x, y])

                if idx % 2 == 0:
                    list_z_ancillas_index.append(qubit_idx)
                elif idx % 2 == 1:
                    list_x_ancillas_index.append(qubit_idx)

                qubit_idx += 1

        for x in range(4, Lx_ancilla, 4):
            circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, Ly_ancilla - 1) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(x, Ly_ancilla - 1): qubit_idx})
            index_to_coordinate.append([x, Ly_ancilla - 1])
            list_x_ancillas_index.append(qubit_idx)

            qubit_idx += 1

        # reference qubit coordinates
        reference_index = qubit_idx
        circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(Lx_ancilla - 1, Ly_ancilla - 1) + " {}".format(qubit_idx))
        coord_to_index.update({"({},{})".format(Lx_ancilla - 1, Ly_ancilla - 1): qubit_idx})
        index_to_coordinate.append([Lx_ancilla - 1, Ly_ancilla - 1])
        qubit_idx += 1

        reference_ancillas = []
        # logical z reference qubit
        for i in range(Ly):
            circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(Lx_ancilla + i, Ly_ancilla - 1) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(Lx_ancilla + i, Ly_ancilla - 1): qubit_idx})
            index_to_coordinate.append([Lx_ancilla + i, Ly_ancilla - 1])
            reference_ancillas.append(qubit_idx)
            qubit_idx += 1

        # logical x reference qubit
        for i in range(Lx):
            circuit.append_from_stim_program_text(
                "QUBIT_COORDS({},{})".format(Lx_ancilla - 1, Ly_ancilla + i) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(Lx_ancilla - 1, Ly_ancilla + i): qubit_idx})
            index_to_coordinate.append([Lx_ancilla - 1, Ly_ancilla + i])
            reference_ancillas.append(qubit_idx)
            qubit_idx += 1

        measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
        measure_x = self.measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index)
        measure_bell = self.measure_bell_stabilizers(coord_to_index, reference_index, reference_ancillas, Ly, Lx)

        circuit.append("R", range(2 * Lx * Ly - 1))
        circuit.append("TICK")

        circuit += measure_z
        circuit += measure_x
        circuit += measure_bell

        circuit.append("MR", list_z_ancillas_index)
        circuit.append("MR", list_x_ancillas_index)
        circuit.append("MR", reference_ancillas)
        circuit.append("TICK")

        # errors
        circuit.append("DEPOLARIZE1", list_data_index, self.noise)
        circuit.append("TICK")

        circuit += measure_z
        circuit += measure_x
        circuit += measure_bell

        circuit.append("M", list_z_ancillas_index)
        circuit.append("M", list_x_ancillas_index)
        circuit.append("M", reference_ancillas)

        offset = (Lx * Ly - 1) // 2
        r_offset = len(reference_ancillas)

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + offset + r_offset,
                                                                                         1 + idx + 3 * offset + 2 * r_offset))

        for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + r_offset, 1 + idx + 2 * offset + 2 * r_offset))

        for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text("DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx, 1 + idx + r_offset + 2 * offset))

        return circuit

    def get_syndromes(self, n, flip: bool = False, supervised: bool = False, only_syndromes: bool = False):
        sampler = self.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=n)
        samples = np.array(list(map(lambda y: np.where(y, 1, 0), samples)))
        syndromes = samples
        if only_syndromes:
            return syndromes[:, :-6]
        return syndromes
