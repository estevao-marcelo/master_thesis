"""
Gerador de sinais de qualidade de energia eletrica para classificacao.
Versao com amostragem 100% ESTRATIFICADA usando Latin Hypercube Sampling.
"""

import numpy as np
import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.stats import qmc

class PowerQualitySignalGenerator:
    """
    Gerador de sinais sinteticos de qualidade de energia eletrica.
    Implementa 16 classes de disturbios com amostragem estratificada.
    """
    
    def __init__(self, 
                 dataset_path: str = "data/power_quality_signals",
                 f: float = 60.0,
                 duration: float = 0.2,
                 samples_per_cycle: int = 128):
        
        self.dataset_path = Path(dataset_path)
        self.f = f
        self.duration = duration
        self.samples_per_cycle = samples_per_cycle
        
        # Calcular parametros temporais
        self.fs = self.samples_per_cycle * self.f
        self.n_samples = int(self.fs * self.duration)
        self.t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        
        self.logger = logging.getLogger('PowerQualityGenerator')
        
        # Definir classes e seus geradores estratificados
        self.signal_classes = {
            'C1_Normal': self._generate_C1_stratified,
            'C2_Sag': self._generate_C2_stratified,
            'C3_Swell': self._generate_C3_stratified,
            'C4_Interruption': self._generate_C4_stratified,
            'C5_Flicker': self._generate_C5_stratified,
            'C6_Transient': self._generate_C6_stratified,
            'C7_Harmonics': self._generate_C7_stratified,
            'C8_Notch': self._generate_C8_stratified,
            'C9_Spike': self._generate_C9_stratified,
            'C10_Sag_Transient': self._generate_C10_stratified,
            'C11_Swell_Transient': self._generate_C11_stratified,
            'C12_Sag_Harmonic': self._generate_C12_stratified,
            'C13_Interruption_Harmonic': self._generate_C13_stratified,
            'C14_Swell_Harmonic': self._generate_C14_stratified,
            'C15_Flicker_Harmonic': self._generate_C15_stratified,
            'C16_Transient_Harmonic': self._generate_C16_stratified
        }
        
        self.logger.info(f'Inicializado gerador de sinais PQ (ESTRATIFICADO)')
        self.logger.info(f'  Frequencia: {self.f} Hz')
        self.logger.info(f'  Duracao: {self.duration} s')
        self.logger.info(f'  Amostras: {self.n_samples}')
        self.logger.info(f'  Fs: {self.fs} Hz')

    def u(self, x):
        """Funcao degrau unitario"""
        return np.heaviside(x, 0)

    # ==================== FUNCOES DE SINAIS ====================
    
    def C1(self, t, A, f):
        """C1: Normal Signal"""
        omega = 2 * np.pi * f
        return A * np.cos(omega * t)

    def C2(self, t, A1, k, t1, t2, f):
        """C2: Sag"""
        omega = 2 * np.pi * f
        return A1 * (1 - k * (self.u(t - t1) - self.u(t - t2))) * np.cos(omega * t)

    def C3(self, t, A1, k, t1, t2, f):
        """C3: Swell"""
        omega = 2 * np.pi * f
        return A1 * (1 + k * (self.u(t - t1) - self.u(t - t2))) * np.cos(omega * t)

    def C4(self, t, A1, k, t1, t2, f):
        """C4: Interruption"""
        omega = 2 * np.pi * f
        return A1 * (1 - k * (self.u(t - t1) - self.u(t - t2))) * np.cos(omega * t)

    def C5(self, t, A, alpha, beta, f):
        """C5: Flicker"""
        omega = 2 * np.pi * f
        return A * (1 + alpha * np.cos(beta * t)) * np.cos(omega * t)

    def C6(self, t, A, f, k, tau, t1, omega_n):
        """C6: Transient"""
        omega = 2 * np.pi * f
        transient_term = np.zeros_like(t)
        mask = t >= t1
        transient_term[mask] = k * np.exp(-(t[mask] - t1) / tau) * np.cos(omega_n * (t[mask] - t1))
        return A * (np.cos(omega * t) + transient_term)

    def C7(self, t, A, f, alpha3, alpha5, alpha7):
        """C7: Harmonics"""
        omega = 2 * np.pi * f
        return (A * np.cos(omega * t) +
                alpha3 * np.cos(3 * omega * t) +
                alpha5 * np.cos(5 * omega * t) +
                alpha7 * np.cos(7 * omega * t))

    def C8(self, t, A, f, K, t1, t2):
        """C8: Notch"""
        omega = 2 * np.pi * f
        sum_term = np.zeros_like(t)
        for n in range(9):
            sum_term += K * (self.u(t - (t1 - 0.02 * n)) - self.u(t - (t2 - 0.02 * n)))
        return A * np.cos(omega * t) - np.sign(np.cos(omega * t)) * sum_term

    def C9(self, t, A, f, K, t1, t2):
        """C9: Spike"""
        omega = 2 * np.pi * f
        sum_term = np.zeros_like(t)
        for n in range(9):
            sum_term += K * (self.u(t - (t1 - 0.02 * n)) - self.u(t - (t2 - 0.02 * n)))
        return A * np.cos(omega * t) + np.sign(np.cos(omega * t)) * sum_term

    def C10(self, t, A1, k, t1, t2, f, A, rho, tau, t3, omega_n):
        """C10: Sag with Transient"""
        omega = 2 * np.pi * f
        transient_term = np.zeros_like(t)
        mask = t >= t3
        transient_term[mask] = A * rho * np.exp(-(t[mask] - t3) / tau) * np.cos(omega_n * (t[mask] - t3))
        return (A1 * (1 - k * (self.u(t - t1) - self.u(t - t2))) * np.cos(omega * t) + 
                transient_term)

    def C11(self, t, A1, k, t1, t2, f, A, rho, tau, t3, omega_n):
        """C11: Swell with Transient"""
        omega = 2 * np.pi * f
        transient_term = np.zeros_like(t)
        mask = t >= t3
        transient_term[mask] = A * rho * np.exp(-(t[mask] - t3) / tau) * np.cos(omega_n * (t[mask] - t3))
        return (A1 * (1 + k * (self.u(t - t1) - self.u(t - t2))) * np.cos(omega * t) + 
                transient_term)

    def C12(self, t, A1, k, t1, t2, f, alpha3, alpha5, alpha7):
        """C12: Sag with Harmonic"""
        omega = 2 * np.pi * f
        return (A1 * (1 - k * (self.u(t - t1) - self.u(t - t2))) * np.cos(omega * t) +
                alpha3 * np.cos(3 * omega * t) +
                alpha5 * np.cos(5 * omega * t) +
                alpha7 * np.cos(7 * omega * t))

    def C13(self, t, A1, k, t1, t2, f, alpha3, alpha5, alpha7):
        """C13: Interruption with Harmonic"""
        omega = 2 * np.pi * f
        return (A1 * (1 - k * (self.u(t - t1) - self.u(t - t2))) * np.cos(omega * t) +
                alpha3 * np.cos(3 * omega * t) +
                alpha5 * np.cos(5 * omega * t) +
                alpha7 * np.cos(7 * omega * t))

    def C14(self, t, A1, k, t1, t2, f, alpha3, alpha5, alpha7):
        """C14: Swell with Harmonic"""
        omega = 2 * np.pi * f
        return (A1 * (1 + k * (self.u(t - t1) - self.u(t - t2))) * np.cos(omega * t) +
                alpha3 * np.cos(3 * omega * t) +
                alpha5 * np.cos(5 * omega * t) +
                alpha7 * np.cos(7 * omega * t))

    def C15(self, t, A, alpha, beta, f, alpha3, alpha5, alpha7):
        """C15: Flicker with Harmonic"""
        omega = 2 * np.pi * f
        return (A * (1 + alpha * np.cos(beta * t)) * np.cos(omega * t) +
                alpha3 * np.cos(3 * omega * t) +
                alpha5 * np.cos(5 * omega * t) +
                alpha7 * np.cos(7 * omega * t))

    def C16(self, t, A, f, k, tau, t1, omega_n, alpha3, alpha5, alpha7):
        """C16: Transient with Harmonic"""
        omega = 2 * np.pi * f
        transient_term = np.zeros_like(t)
        mask = t >= t1
        transient_term[mask] = k * np.exp(-(t[mask] - t1) / tau) * np.cos(omega_n * (t[mask] - t1))
        return ((A * (np.cos(omega * t) + transient_term)) +
                alpha3 * np.cos(3 * omega * t) +
                alpha5 * np.cos(5 * omega * t) +
                alpha7 * np.cos(7 * omega * t))

    # ==================== GERADORES ESTRATIFICADOS ====================

    def _generate_C1_stratified(self, n_samples: int) -> List[Dict]:
        """C1: Normal - Sem parametros variaveis"""
        return [{'A': 1.0} for _ in range(n_samples)]

    def _generate_C2_stratified(self, n_samples: int) -> List[Dict]:
        """C2: Sag - 3 parametros: k, duracao, t1"""
        T = 1 / self.f
        
        # Latin Hypercube Sampling com 3 dimensoes
        sampler = qmc.LatinHypercube(d=3, seed=42)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            # Parametro 1: k (profundidade do sag)
            k = sample[0] * (0.9 - 0.1) + 0.1
            
            # Parametro 2: duracao (T a 9T)
            duration = sample[1] * (9*T - T) + T
            
            # Parametro 3: t1 (posicao inicial)
            t1_max = self.duration - duration
            t1 = sample[2] * t1_max if t1_max > 0 else 0
            t2 = t1 + duration
            
            params_list.append({
                'A1': 1.0,
                'k': k,
                't1': t1,
                't2': t2
            })
        
        return params_list

    def _generate_C3_stratified(self, n_samples: int) -> List[Dict]:
        """C3: Swell - 3 parametros: k, duracao, t1"""
        T = 1 / self.f
        
        sampler = qmc.LatinHypercube(d=3, seed=43)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            k = sample[0] * (0.9 - 0.1) + 0.1
            duration = sample[1] * (9*T - T) + T
            t1_max = self.duration - duration
            t1 = sample[2] * t1_max if t1_max > 0 else 0
            t2 = t1 + duration
            
            params_list.append({
                'A1': 1.0,
                'k': k,
                't1': t1,
                't2': t2
            })
        
        return params_list

    def _generate_C4_stratified(self, n_samples: int) -> List[Dict]:
        """C4: Interruption - 3 parametros: k, duracao, t1"""
        T = 1 / self.f
        
        sampler = qmc.LatinHypercube(d=3, seed=44)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            k = sample[0] * (0.99 - 0.9) + 0.9
            duration = sample[1] * (9*T - T) + T
            t1_max = self.duration - duration
            t1 = sample[2] * t1_max if t1_max > 0 else 0
            t2 = t1 + duration
            
            params_list.append({
                'A1': 1.0,
                'k': k,
                't1': t1,
                't2': t2
            })
        
        return params_list

    def _generate_C5_stratified(self, n_samples: int) -> List[Dict]:
        """C5: Flicker - 2 parametros: alpha, beta"""
        sampler = qmc.LatinHypercube(d=2, seed=45)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            alpha = sample[0] * (0.2 - 0.1) + 0.1
            f_flicker = sample[1] * (20 - 5) + 5
            beta = 2 * np.pi * f_flicker
            
            params_list.append({
                'A': 1.0,
                'alpha': alpha,
                'beta': beta
            })
        
        return params_list

    def _generate_C6_stratified(self, n_samples: int) -> List[Dict]:
        """C6: Transient - 4 parametros: k, tau, t1, omega_n"""
        sampler = qmc.LatinHypercube(d=4, seed=46)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            k = sample[0] * (0.8 - 0.1) + 0.1
            tau = sample[1] * (1000e-6 - 150e-6) + 150e-6
            t1 = sample[2] * (self.duration - 0.05 - 0.01) + 0.01
            f_n = sample[3] * (1600 - 700) + 700
            omega_n = 2 * np.pi * f_n
            
            params_list.append({
                'A': 1.0,
                'k': k,
                'tau': tau,
                't1': t1,
                'omega_n': omega_n
            })
        
        return params_list

    def _generate_C7_stratified(self, n_samples: int) -> List[Dict]:
        """C7: Harmonics - 3 parametros: alpha3, alpha5, alpha7"""
        sampler = qmc.LatinHypercube(d=3, seed=47)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            alpha3 = sample[0] * (0.1 - 0.02) + 0.02
            alpha5 = sample[1] * (0.1 - 0.02) + 0.02
            alpha7 = sample[2] * (0.1 - 0.02) + 0.02
            
            params_list.append({
                'A': 1.0,
                'alpha3': alpha3,
                'alpha5': alpha5,
                'alpha7': alpha7
            })
        
        return params_list

    def _generate_C8_stratified(self, n_samples: int) -> List[Dict]:
        """C8: Notch - 3 parametros: K, duracao, t1"""
        sampler = qmc.LatinHypercube(d=3, seed=48)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            K = sample[0] * (0.4 - 0.1) + 0.1
            duration = sample[1] * (0.02 - 0.001) + 0.001
            t1_max = self.duration - 0.18
            if t1_max <= 0.01:
                t1_max = 0.01
                t1 = 0.01
            else:
                t1 = sample[2] * (t1_max - 0.01) + 0.01
            t2 = t1 + duration
            
            params_list.append({
                'A': 1.0,
                'K': K,
                't1': t1,
                't2': t2
            })
        
        return params_list

    def _generate_C9_stratified(self, n_samples: int) -> List[Dict]:
        """C9: Spike - 3 parametros: K, duracao, t1"""
        sampler = qmc.LatinHypercube(d=3, seed=49)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            K = sample[0] * (0.4 - 0.1) + 0.1
            duration = sample[1] * (0.02 - 0.001) + 0.001
            t1_max = self.duration - 0.18
            if t1_max <= 0.01:
                t1_max = 0.01
                t1 = 0.01
            else:
                t1 = sample[2] * (t1_max - 0.01) + 0.01
            t2 = t1 + duration
            
            params_list.append({
                'A': 1.0,
                'K': K,
                't1': t1,
                't2': t2
            })
        
        return params_list

    def _generate_C10_stratified(self, n_samples: int) -> List[Dict]:
        """C10: Sag with Transient - 7 parametros (removido sample[7] nao usado)"""
        T = 1 / self.f
        
        sampler = qmc.LatinHypercube(d=7, seed=50)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            # Sag params
            k = sample[0] * (0.9 - 0.1) + 0.1
            duration = sample[1] * (9*T - T) + T
            t1_max = self.duration - duration
            t1 = sample[2] * t1_max if t1_max > 0 else 0
            t2 = t1 + duration
            
            # Transient params
            rho = sample[3] * (0.8 - 0.1) + 0.1
            tau = sample[4] * (1000e-6 - 150e-6) + 150e-6
            t3 = sample[5] * (self.duration - 0.05 - 0.01) + 0.01
            f_n = sample[6] * (1600 - 700) + 700
            omega_n = 2 * np.pi * f_n
            
            params_list.append({
                'A1': 1.0,
                'k': k,
                't1': t1,
                't2': t2,
                'A': 1.0,
                'rho': rho,
                'tau': tau,
                't3': t3,
                'omega_n': omega_n
            })
        
        return params_list

    def _generate_C11_stratified(self, n_samples: int) -> List[Dict]:
        """C11: Swell with Transient - 7 parametros (removido sample[7] nao usado)"""
        T = 1 / self.f
        
        sampler = qmc.LatinHypercube(d=7, seed=51)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            # Swell params
            k = sample[0] * (0.9 - 0.1) + 0.1
            duration = sample[1] * (9*T - T) + T
            t1_max = self.duration - duration
            t1 = sample[2] * t1_max if t1_max > 0 else 0
            t2 = t1 + duration
            
            # Transient params
            rho = sample[3] * (0.8 - 0.1) + 0.1
            tau = sample[4] * (1000e-6 - 150e-6) + 150e-6
            t3 = sample[5] * (self.duration - 0.05 - 0.01) + 0.01
            f_n = sample[6] * (1600 - 700) + 700
            omega_n = 2 * np.pi * f_n
            
            params_list.append({
                'A1': 1.0,
                'k': k,
                't1': t1,
                't2': t2,
                'A': 1.0,
                'rho': rho,
                'tau': tau,
                't3': t3,
                'omega_n': omega_n
            })
        
        return params_list

    def _generate_C12_stratified(self, n_samples: int) -> List[Dict]:
        """C12: Sag with Harmonic - 6 parametros"""
        T = 1 / self.f
        
        sampler = qmc.LatinHypercube(d=6, seed=52)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            k = sample[0] * (0.9 - 0.1) + 0.1
            duration = sample[1] * (9*T - T) + T
            t1_max = self.duration - duration
            t1 = sample[2] * t1_max if t1_max > 0 else 0
            t2 = t1 + duration
            alpha3 = sample[3] * (0.1 - 0.02) + 0.02
            alpha5 = sample[4] * (0.1 - 0.02) + 0.02
            alpha7 = sample[5] * (0.1 - 0.02) + 0.02
            
            params_list.append({
                'A1': 1.0,
                'k': k,
                't1': t1,
                't2': t2,
                'alpha3': alpha3,
                'alpha5': alpha5,
                'alpha7': alpha7
            })
        
        return params_list

    def _generate_C13_stratified(self, n_samples: int) -> List[Dict]:
        """C13: Interruption with Harmonic - 6 parametros"""
        T = 1 / self.f
        
        sampler = qmc.LatinHypercube(d=6, seed=53)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            k = sample[0] * (0.99 - 0.9) + 0.9
            duration = sample[1] * (9*T - T) + T
            t1_max = self.duration - duration
            t1 = sample[2] * t1_max if t1_max > 0 else 0
            t2 = t1 + duration
            alpha3 = sample[3] * (0.1 - 0.02) + 0.02
            alpha5 = sample[4] * (0.1 - 0.02) + 0.02
            alpha7 = sample[5] * (0.1 - 0.02) + 0.02
            
            params_list.append({
                'A1': 1.0,
                'k': k,
                't1': t1,
                't2': t2,
                'alpha3': alpha3,
                'alpha5': alpha5,
                'alpha7': alpha7
            })
        
        return params_list

    def _generate_C14_stratified(self, n_samples: int) -> List[Dict]:
        """C14: Swell with Harmonic - 6 parametros"""
        T = 1 / self.f
        
        sampler = qmc.LatinHypercube(d=6, seed=54)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            k = sample[0] * (0.9 - 0.1) + 0.1
            duration = sample[1] * (9*T - T) + T
            t1_max = self.duration - duration
            t1 = sample[2] * t1_max if t1_max > 0 else 0
            t2 = t1 + duration
            alpha3 = sample[3] * (0.1 - 0.02) + 0.02
            alpha5 = sample[4] * (0.1 - 0.02) + 0.02
            alpha7 = sample[5] * (0.1 - 0.02) + 0.02
            
            params_list.append({
                'A1': 1.0,
                'k': k,
                't1': t1,
                't2': t2,
                'alpha3': alpha3,
                'alpha5': alpha5,
                'alpha7': alpha7
            })
        
        return params_list

    def _generate_C15_stratified(self, n_samples: int) -> List[Dict]:
        """C15: Flicker with Harmonic - 5 parametros"""
        sampler = qmc.LatinHypercube(d=5, seed=55)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            alpha = sample[0] * (0.2 - 0.1) + 0.1
            f_flicker = sample[1] * (20 - 5) + 5
            beta = 2 * np.pi * f_flicker
            alpha3 = sample[2] * (0.1 - 0.02) + 0.02
            alpha5 = sample[3] * (0.1 - 0.02) + 0.02
            alpha7 = sample[4] * (0.1 - 0.02) + 0.02
            
            params_list.append({
                'A': 1.0,
                'alpha': alpha,
                'beta': beta,
                'alpha3': alpha3,
                'alpha5': alpha5,
                'alpha7': alpha7
            })
        
        return params_list

    def _generate_C16_stratified(self, n_samples: int) -> List[Dict]:
        """C16: Transient with Harmonic - 7 parametros"""
        sampler = qmc.LatinHypercube(d=7, seed=56)
        samples = sampler.random(n=n_samples)
        
        params_list = []
        for sample in samples:
            k = sample[0] * (0.8 - 0.1) + 0.1
            tau = sample[1] * (1000e-6 - 150e-6) + 150e-6
            t1 = sample[2] * (self.duration - 0.05 - 0.01) + 0.01
            f_n = sample[3] * (1600 - 700) + 700
            omega_n = 2 * np.pi * f_n
            alpha3 = sample[4] * (0.1 - 0.02) + 0.02
            alpha5 = sample[5] * (0.1 - 0.02) + 0.02
            alpha7 = sample[6] * (0.1 - 0.02) + 0.02
            
            params_list.append({
                'A': 1.0,
                'k': k,
                'tau': tau,
                't1': t1,
                'omega_n': omega_n,
                'alpha3': alpha3,
                'alpha5': alpha5,
                'alpha7': alpha7
            })
        
        return params_list

    def generate_single_signal(self, signal_class: str, params: Dict) -> np.ndarray:
        """Gera um unico sinal da classe especificada."""
        params['f'] = self.f
        signal_func = getattr(self, signal_class.split('_')[0])
        signal = signal_func(self.t, **params)
        
        # Adicionar ruido minimo
        noise_level = 0.001
        signal += np.random.normal(0, noise_level, signal.shape)
        
        return signal

    def generate_dataset(self, n_samples_per_class: int = 300, save: bool = True) -> Dict:
        """
        Gera dataset completo com amostragem ESTRATIFICADA.
        
        Args:
            n_samples_per_class: Numero de amostras por classe
            save: Salvar dataset em disco (formato pickle)
            
        Returns:
            Dict com sinais e labels
        """
        self.logger.info(f'Gerando dataset ESTRATIFICADO com {n_samples_per_class} amostras/classe')
        
        all_signals = []
        all_labels = []
        all_params = []  # Guardar parametros para analise posterior
        class_names = list(self.signal_classes.keys())
        
        for class_idx, signal_class in enumerate(class_names):
            self.logger.info(f'Gerando classe {signal_class} ({class_idx+1}/{len(class_names)})')
            
            # Gerar parametros estratificados para toda a classe de uma vez
            stratified_params = self.signal_classes[signal_class](n_samples_per_class)
            
            class_signals = []
            class_params = []
            for i, params in enumerate(stratified_params):
                signal = self.generate_single_signal(signal_class, params.copy())
                class_signals.append(signal)
                class_params.append(params)
                all_labels.append(class_idx)
                
                if (i + 1) % 50 == 0:
                    self.logger.info(f'  {i+1}/{n_samples_per_class} amostras geradas')
            
            all_signals.extend(class_signals)
            all_params.extend(class_params)
            
            # Salvar classe separadamente se solicitado
            if save:
                class_dir = self.dataset_path / signal_class
                class_dir.mkdir(parents=True, exist_ok=True)
                
                for i, signal in enumerate(class_signals):
                    signal_path = class_dir / f'signal_{i:04d}.npy'
                    np.save(signal_path, signal)
        
        dataset = {
            'signals': np.array(all_signals),
            'labels': np.array(all_labels),
            'params': all_params,  # Parametros usados para gerar cada sinal
            'class_names': class_names,
            'metadata': {
                'fs': self.fs,
                'duration': self.duration,
                'f_nominal': self.f,
                'n_samples': self.n_samples,
                'n_classes': len(class_names),
                'samples_per_class': n_samples_per_class,
                'sampling_method': 'stratified_lhs',
                'total_samples': len(all_signals)
            }
        }
        
        if save:
            # Criar diretorio se nao existir
            self.dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Salvar dataset completo em formato pickle
            dataset_file = self.dataset_path / 'complete_dataset_stratified.pkl'
            with open(dataset_file, 'wb') as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f'Dataset salvo em {dataset_file}')
            self.logger.info(f'  Tamanho do arquivo: {dataset_file.stat().st_size / (1024*1024):.2f} MB')
        
        self.logger.info(f'Dataset gerado: {len(all_signals)} sinais, {len(class_names)} classes')
        return dataset

    def load_dataset(self) -> Dict:
        """Carrega dataset existente do arquivo pickle."""
        dataset_file = self.dataset_path / 'complete_dataset_stratified.pkl'
        
        if dataset_file.exists():
            with open(dataset_file, 'rb') as f:
                dataset = pickle.load(f)
            self.logger.info(f'Dataset carregado: {len(dataset["signals"])} sinais')
            self.logger.info(f'  Metodo: {dataset["metadata"]["sampling_method"]}')
            return dataset
        else:
            raise FileNotFoundError(f'Dataset nao encontrado em {dataset_file}')

    def dataset_exists(self) -> bool:
        """Verifica se o dataset ja existe."""
        dataset_file = self.dataset_path / 'complete_dataset_stratified.pkl'
        return dataset_file.exists()

    def plot_sample_signals(self, n_samples: int = 3, save_plots: bool = True):
        """Plota amostras de cada classe de sinal."""
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        class_names = list(self.signal_classes.keys())
        
        for i, signal_class in enumerate(class_names):
            if i >= 16:
                break
            
            # Gerar parametros estratificados e pegar primeiro
            params_list = self.signal_classes[signal_class](1)
            signal = self.generate_single_signal(signal_class, params_list[0])
            
            axes[i].plot(self.t * 1000, signal, 'b-', linewidth=1)
            axes[i].set_title(f'{signal_class}', fontsize=10)
            axes[i].set_xlabel('Tempo (ms)')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_dir = self.dataset_path / 'plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_dir / 'sample_signals_stratified.png', dpi=300, bbox_inches='tight')
        
        plt.show()

    def analyze_parameter_coverage(self, n_samples: int = 300):
        """Analisa a cobertura dos parametros no espaco estratificado."""
        import matplotlib.pyplot as plt
        
        # Exemplo: analisar C2 (Sag)
        params_list = self._generate_C2_stratified(n_samples)
        
        k_values = [p['k'] for p in params_list]
        duration_values = [(p['t2'] - p['t1']) * 1000 for p in params_list]  # ms
        t1_values = [p['t1'] * 1000 for p in params_list]  # ms
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(k_values, bins=30, edgecolor='black')
        axes[0].set_title('Distribuicao de k (profundidade)')
        axes[0].set_xlabel('k')
        axes[0].set_ylabel('Frequencia')
        
        axes[1].hist(duration_values, bins=30, edgecolor='black')
        axes[1].set_title('Distribuicao de duracao')
        axes[1].set_xlabel('Duracao (ms)')
        axes[1].set_ylabel('Frequencia')
        
        axes[2].scatter(k_values, duration_values, alpha=0.5)
        axes[2].set_title('k vs Duracao (Latin Hypercube)')
        axes[2].set_xlabel('k')
        axes[2].set_ylabel('Duracao (ms)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nAnalise de cobertura (C2 - Sag):")
        print(f"  k: [{min(k_values):.3f}, {max(k_values):.3f}]")
        print(f"  Duracao: [{min(duration_values):.3f}, {max(duration_values):.3f}] ms")
        print(f"  t1: [{min(t1_values):.3f}, {max(t1_values):.3f}] ms")


def create_power_quality_dataset(force_regenerate: bool = False) -> Dict:
    """
    Funcao principal para criar/carregar dataset ESTRATIFICADO.
    """
    logging.basicConfig(level=logging.INFO)
    generator = PowerQualitySignalGenerator()
    
    if generator.dataset_exists() and not force_regenerate:
        print("Dataset ja existe. Carregando...")
        return generator.load_dataset()
    else:
        print("Gerando novo dataset ESTRATIFICADO...")
        return generator.generate_dataset(n_samples_per_class=300)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    generator = PowerQualitySignalGenerator()
    
    # Gerar dataset se nao existir
    if not generator.dataset_exists():
        print("\n=== GERANDO DATASET ESTRATIFICADO ===")
        dataset = generator.generate_dataset(n_samples_per_class=300)
        print(f"\nDataset gerado com {len(dataset['signals'])} sinais")
        print(f"Metodo de amostragem: {dataset['metadata']['sampling_method']}")
    else:
        print("\nDataset ja existe!")
        dataset = generator.load_dataset()
        print(f"Dataset carregado com {len(dataset['signals'])} sinais")
    
    # Plotar amostras
    print("\n=== GERANDO GRAFICOS ===")
    generator.plot_sample_signals()
    
    # Analisar cobertura de parametros
    print("\n=== ANALISANDO COBERTURA DE PARAMETROS ===")
    generator.analyze_parameter_coverage(n_samples=300)