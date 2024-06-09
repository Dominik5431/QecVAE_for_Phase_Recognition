from abc import ABC, abstractmethod
import stim
import numpy as np
from pathlib import Path
from random import random


class QECCode(ABC):
    def __init__(self, distance, noise, random_flip):
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
    def get_syndromes(self, n):
        raise NotImplementedError("Subclasses should implement this!")


class BitFlipSurfaceCode(QECCode):
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

    def get_syndromes(self, n):
        sampler = self.circuit.compile_detector_sampler()
        syndromes = sampler.sample(shots=n)
        # Add here already the last redundant syndrome to have square shape
        z_add = 1
        x_add = 1
        syndromes = np.array(list(map(lambda y: np.where(y, -1, 1), syndromes)))  # tried on 08.05.
        syndromes_final = []
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
            syndromes_final.append(syndrome_shot)
        if self.random_flip:
            syndromes_final = np.array(list(map(lambda y: y if random() < 0.5 else -y, syndromes_final)))
        return syndromes_final

class DepolarizingSurfaceCode(QECCode):
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
            circuit.append("DEPOLARIZE1", [i], self.noise)
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

    def get_syndromes(self, n):
        sampler = self.circuit.compile_detector_sampler()
        syndromes = sampler.sample(shots=n)
        # Add here already the last redundant syndrom to have square shape
        z_add = 1
        x_add = 1
        syndromes = np.array(list(map(lambda y: np.where(y, -1, 1), syndromes)))  # tried on 08.05.
        syndromes_final = []
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
            syndromes_final.append(syndrome_shot)
        if self.random_flip:
            syndromes_final = np.array(list(map(lambda y: y if random() < 0.5 else -y, syndromes_final)))
        return syndromes_final


class SurfaceCodePheno(QECCode):
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


class RotatedSurfaceCode(QECCode):
    def __init__(self, distance, noise, random_flip):
        super().__init__(distance, noise, random_flip)
        self.syndromes = 2 * int((distance ** 2 - 1) / 2)

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
        # Measure all stabilizers to project into eigenstate of stabilizers, save outcome
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
        syndromes = sampler.sample(shots=n)  # shape n x 2*syndrom
        syndromes = list(map(lambda x: np.where(x, -1, 1), syndromes))
        syndromes_result = []
        for elm in syndromes:
            append = elm[:int(self.syndromes)] * elm[int(self.syndromes):]
            syndromes_result.append(append)
        return syndromes_result
