"""
Gerador de sinais de qualidade de energia eletrica para classificacao.
Baseado nas 16 classes de disturbios definidas na literatura tecnica.
"""

import numpy as np
import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class PowerQualitySignalGenerator:
    """
    Gerador de sinais sinteticos de qualidade de energia eletrica.
    Implementa 16 classes de disturbios com parametros variaveis.
    """
    
    def __init__(self, 
                 dataset_path: str = "data/power_quality_signals",
                 f: float = 60.0,  # Frequencia nominal (Hz)
                 duration: float = 0.2,  # Duracao do sinal (s) 
                 samples_per_cycle: int = 256):  # Amostras por ciclo
        
        self.dataset_path = Path(dataset_path)
        self.f = f  # 60 Hz
        self.duration = duration  # 0.2 segundos
        self.samples_per_cycle = samples_per_cycle
        
        # Calcular parametros temporais
        self.fs = self.samples_per_cycle * self.f  # Frequencia de amostragem
        self.n_samples = int(self.fs * self.duration)  # Total de amostras
        self.t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        
        self.logger = logging.getLogger('PowerQualityGenerator')
        
        # Definir classes e parametros
        self.signal_classes = {
            'C1_Normal': self._generate_C1_params,
            'C2_Sag': self._generate_C2_params,
            'C3_Swell': self._generate_C3_params,
            'C4_Interruption': self._generate_C4_params,
            'C5_Flicker': self._generate_C5_params,
            'C6_Transient': self._generate_C6_params,
            'C7_Harmonics': self._generate_C7_params,
            'C8_Notch': self._generate_C8_params,
            'C9_Spike': self._generate_C9_params,
            'C10_Sag_Transient': self._generate_C10_params,
            'C11_Swell_Transient': self._generate_C11_params,
            'C12_Sag_Harmonic': self._generate_C12_params,
            'C13_Interruption_Harmonic': self._generate_C13_params,
            'C14_Swell_Harmonic': self._generate_C14_params,
            'C15_Flicker_Harmonic': self._generate_C15_params,
            'C16_Transient_Harmonic': self._generate_C16_params
        }
        
        self.logger.info(f'Inicializado gerador de sinais PQ')
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

    # ==================== GERADORES DE PARAMETROS ====================

    def _generate_C1_params(self) -> Dict:
        """Parametros para C1: Normal Signal"""
        return {'A': 1.0}

    def _generate_C2_params(self) -> Dict:
        """Parametros para C2: Sag"""
        T = 1 / self.f
        
        # Primeiro escolhe a duração desejada (T <= t2-t1 <= 9T)
        min_duration = T
        max_duration = 9 * T
        duration = np.random.uniform(min_duration, max_duration)
        
        # Depois posiciona a janela
        t1_max = self.duration - duration
        t1 = np.random.uniform(0, t1_max)
        t2 = t1 + duration
        
        return {
            'A1': 1.0,
            'k': np.random.uniform(0.1, 0.9),
            't1': t1,
            't2': t2
        }

    def _generate_C3_params(self) -> Dict:
        """Parametros para C3: Swell"""
        T = 1 / self.f
        
        # Primeiro escolhe a duração desejada (T <= t2-t1 <= 9T)
        min_duration = T
        max_duration = 9 * T
        duration = np.random.uniform(min_duration, max_duration)
        
        # Depois posiciona a janela
        t1_max = self.duration - duration
        t1 = np.random.uniform(0, t1_max)
        t2 = t1 + duration
        
        return {
            'A1': 1.0,
            'k': np.random.uniform(0.1, 0.9),
            't1': t1,
            't2': t2
        }

    def _generate_C4_params(self) -> Dict:
        """Parametros para C4: Interruption"""
        T = 1 / self.f
        
        # Primeiro escolhe a duração desejada (T <= t2-t1 <= 9T)
        min_duration = T
        max_duration = 9 * T
        duration = np.random.uniform(min_duration, max_duration)
        
        # Depois posiciona a janela
        t1_max = self.duration - duration
        t1 = np.random.uniform(0, t1_max)
        t2 = t1 + duration
        
        return {
            'A1': 1.0,
            'k': np.random.uniform(0.9, 1.0),
            't1': t1,
            't2': t2
        }

    def _generate_C5_params(self) -> Dict:
        """Parametros para C5: Flicker"""
        # Flicker tipicamente ocorre entre 5-20 Hz (perceptível ao olho humano)
        # beta em rad/s = 2*pi*f_flicker
        return {
            'A': 1.0,
            'alpha': np.random.uniform(0.1, 0.2),
            'beta': 2 * np.pi * np.random.uniform(5, 20)  # 5-20 Hz
        }

    def _generate_C6_params(self) -> Dict:
        """Parametros para C6: Transient"""
        return {
            'A': 1.0,
            'k': np.random.uniform(0.1, 0.8),
            'tau': np.random.uniform(150e-6, 1000e-6),  # Converter para segundos
            't1': np.random.uniform(0.01, self.duration - 0.05),
            'omega_n': 2 * np.pi * np.random.uniform(700, 1600)
        }

    def _generate_C7_params(self) -> Dict:
        """Parametros para C7: Harmonics"""
        return {
            'A': 1.0,
            'alpha3': np.random.uniform(0.02, 0.1),
            'alpha5': np.random.uniform(0.02, 0.1),
            'alpha7': np.random.uniform(0.02, 0.1)
        }

    def _generate_C8_params(self) -> Dict:
        """Parametros para C8: Notch"""
        # Notches têm duração muito curta (não seguem a regra 0.5T-9T)
        min_duration = 0.001  # 1ms
        max_duration = 0.02   # 20ms
        duration = np.random.uniform(min_duration, max_duration)
        
        # Posiciona a janela garantindo espaço para os 9 notches repetidos
        # Cada notch é repetido a cada 0.02s, então precisa de ~0.18s no total
        t1_max = max(0.01, self.duration - 0.18)
        t1 = np.random.uniform(0.01, t1_max)
        t2 = t1 + duration
        
        return {
            'A': 1.0,
            'K': np.random.uniform(0.1, 0.4),
            't1': t1,
            't2': t2
        }

    def _generate_C9_params(self) -> Dict:
        """Parametros para C9: Spike"""
        # Spikes têm duração muito curta (não seguem a regra 0.5T-9T)
        min_duration = 0.001  # 1ms
        max_duration = 0.02   # 20ms
        duration = np.random.uniform(min_duration, max_duration)
        
        # Posiciona a janela garantindo espaço para os 9 spikes repetidos
        # Cada spike é repetido a cada 0.02s, então precisa de ~0.18s no total
        t1_max = max(0.01, self.duration - 0.18)
        t1 = np.random.uniform(0.01, t1_max)
        t2 = t1 + duration
        
        return {
            'A': 1.0,
            'K': np.random.uniform(0.1, 0.4),
            't1': t1,
            't2': t2
        }

    def _generate_C10_params(self) -> Dict:
        """Parametros para C10: Sag with Transient"""
        T = 1 / self.f
        
        # Primeiro escolhe a duração desejada para o sag (T <= t2-t1 <= 9T)
        min_duration = T
        max_duration = 9 * T
        duration = np.random.uniform(min_duration, max_duration)
        
        # Depois posiciona a janela
        t1_max = self.duration - duration
        t1 = np.random.uniform(0, t1_max)
        t2 = t1 + duration
        
        t3 = np.random.uniform(0.01, self.duration - 0.05)
        return {
            'A1': 1.0,
            'k': np.random.uniform(0.1, 0.9),
            't1': t1,
            't2': t2,
            'A': 1.0,
            'rho': np.random.uniform(0.1, 0.8),
            'tau': np.random.uniform(150e-6, 1000e-6),
            't3': t3,
            'omega_n': 2 * np.pi * np.random.uniform(700, 1600)
        }

    def _generate_C11_params(self) -> Dict:
        """Parametros para C11: Swell with Transient"""
        T = 1 / self.f
        
        # Primeiro escolhe a duração desejada para o swell (T <= t2-t1 <= 9T)
        min_duration = T
        max_duration = 9 * T
        duration = np.random.uniform(min_duration, max_duration)
        
        # Depois posiciona a janela
        t1_max = self.duration - duration
        t1 = np.random.uniform(0, t1_max)
        t2 = t1 + duration
        
        t3 = np.random.uniform(0.01, self.duration - 0.05)
        return {
            'A1': 1.0,
            'k': np.random.uniform(0.1, 0.9),
            't1': t1,
            't2': t2,
            'A': 1.0,
            'rho': np.random.uniform(0.1, 0.8),
            'tau': np.random.uniform(150e-6, 1000e-6),
            't3': t3,
            'omega_n': 2 * np.pi * np.random.uniform(700, 1600)
        }

    def _generate_C12_params(self) -> Dict:
        """Parametros para C12: Sag with Harmonic"""
        T = 1 / self.f
        
        # Primeiro escolhe a duração desejada para o sag (T <= t2-t1 <= 9T)
        min_duration = T
        max_duration = 9 * T
        duration = np.random.uniform(min_duration, max_duration)
        
        # Depois posiciona a janela
        t1_max = self.duration - duration
        t1 = np.random.uniform(0, t1_max)
        t2 = t1 + duration
        
        return {
            'A1': 1.0,
            'k': np.random.uniform(0.1, 0.9),
            't1': t1,
            't2': t2,
            'alpha3': np.random.uniform(0.02, 0.1),
            'alpha5': np.random.uniform(0.02, 0.1),
            'alpha7': np.random.uniform(0.02, 0.1)
        }

    def _generate_C13_params(self) -> Dict:
        """Parametros para C13: Interruption with Harmonic"""
        T = 1 / self.f
        
        # Primeiro escolhe a duração desejada para a interrupção (T <= t2-t1 <= 9T)
        min_duration = T
        max_duration = 9 * T
        duration = np.random.uniform(min_duration, max_duration)
        
        # Depois posiciona a janela
        t1_max = self.duration - duration
        t1 = np.random.uniform(0, t1_max)
        t2 = t1 + duration
        
        return {
            'A1': 1.0,
            'k': np.random.uniform(0.9, 1.0),
            't1': t1,
            't2': t2,
            'alpha3': np.random.uniform(0.02, 0.1),
            'alpha5': np.random.uniform(0.02, 0.1),
            'alpha7': np.random.uniform(0.02, 0.1)
        }

    def _generate_C14_params(self) -> Dict:
        """Parametros para C14: Swell with Harmonic"""
        T = 1 / self.f
        
        # Primeiro escolhe a duração desejada para o swell (T <= t2-t1 <= 9T)
        min_duration = T
        max_duration = 9 * T
        duration = np.random.uniform(min_duration, max_duration)
        
        # Depois posiciona a janela
        t1_max = self.duration - duration
        t1 = np.random.uniform(0, t1_max)
        t2 = t1 + duration
        
        return {
            'A1': 1.0,
            'k': np.random.uniform(0.1, 0.9),
            't1': t1,
            't2': t2,
            'alpha3': np.random.uniform(0.02, 0.1),
            'alpha5': np.random.uniform(0.02, 0.1),
            'alpha7': np.random.uniform(0.02, 0.1)
        }

    def _generate_C15_params(self) -> Dict:
        """Parametros para C15: Flicker with Harmonic"""
        # Flicker tipicamente ocorre entre 5-20 Hz (perceptível ao olho humano)
        return {
            'A': 1.0,
            'alpha': np.random.uniform(0.1, 0.2),
            'beta': 2 * np.pi * np.random.uniform(5, 20),  # 5-20 Hz
            'alpha3': np.random.uniform(0.02, 0.1),
            'alpha5': np.random.uniform(0.02, 0.1),
            'alpha7': np.random.uniform(0.02, 0.1)
        }

    def _generate_C16_params(self) -> Dict:
        """Parametros para C16: Transient with Harmonic"""
        return {
            'A': 1.0,
            'k': np.random.uniform(0.1, 0.8),
            'tau': np.random.uniform(150e-6, 1000e-6),
            't1': np.random.uniform(0.01, self.duration - 0.05),
            'omega_n': 2 * np.pi * np.random.uniform(700, 1600),
            'alpha3': np.random.uniform(0.02, 0.1),
            'alpha5': np.random.uniform(0.02, 0.1),
            'alpha7': np.random.uniform(0.02, 0.1)
        }

    def generate_single_signal(self, signal_class: str, params: Dict = None) -> np.ndarray:
        """
        Gera um unico sinal da classe especificada.
        """
        if params is None:
            params = self.signal_classes[signal_class]()
        
        # Adicionar frequencia e tempo aos parametros
        params['f'] = self.f
        
        # Chamar funcao correspondente
        signal_func = getattr(self, signal_class.split('_')[0])
        signal = signal_func(self.t, **params)
        
        # Adicionar ruido minimo
        noise_level = 0.001
        signal += np.random.normal(0, noise_level, signal.shape)
        
        return signal

    def generate_dataset(self, n_samples_per_class: int = 300, save: bool = True) -> Dict:
        """
        Gera dataset completo com todas as classes de sinais.
        
        Args:
            n_samples_per_class: Numero de amostras por classe (default: 300)
            save: Salvar dataset em disco (default: True)
            
        Returns:
            Dict com sinais e labels
        """
        self.logger.info(f'Gerando dataset com {n_samples_per_class} amostras por classe')
        
        all_signals = []
        all_labels = []
        class_names = list(self.signal_classes.keys())
        
        for class_idx, signal_class in enumerate(class_names):
            self.logger.info(f'Gerando classe {signal_class} ({class_idx+1}/{len(class_names)})')
            
            class_signals = []
            for i in range(n_samples_per_class):
                signal = self.generate_single_signal(signal_class)
                class_signals.append(signal)
                all_labels.append(class_idx)
                
                if (i + 1) % 50 == 0:
                    self.logger.info(f'  {i+1}/{n_samples_per_class} amostras geradas')
            
            all_signals.extend(class_signals)
            
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
            'class_names': class_names,
            'metadata': {
                'fs': self.fs,
                'duration': self.duration,
                'f_nominal': self.f,
                'n_samples': self.n_samples,
                'n_classes': len(class_names),
                'samples_per_class': n_samples_per_class
            }
        }
        
        if save:
            # Salvar dataset completo
            dataset_file = self.dataset_path / 'complete_dataset.pkl'
            with open(dataset_file, 'wb') as f:
                pickle.dump(dataset, f)
                
            self.logger.info(f'Dataset salvo em {self.dataset_path}')
        
        self.logger.info(f'Dataset gerado: {len(all_signals)} sinais, {len(class_names)} classes')
        return dataset

    def load_dataset(self) -> Dict:
        """
        Carrega dataset existente.
        """
        dataset_file = self.dataset_path / 'complete_dataset.pkl'
        
        if dataset_file.exists():
            with open(dataset_file, 'rb') as f:
                dataset = pickle.load(f)
            self.logger.info(f'Dataset carregado: {len(dataset["signals"])} sinais')
            return dataset
        else:
            raise FileNotFoundError(f'Dataset nao encontrado em {dataset_file}')

    def dataset_exists(self) -> bool:
        """
        Verifica se o dataset ja existe.
        """
        dataset_file = self.dataset_path / 'complete_dataset.pkl'
        return dataset_file.exists()

    def plot_sample_signals(self, n_samples: int = 3, save_plots: bool = True):
        """
        Plota amostras de cada classe de sinal.
        """
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        class_names = list(self.signal_classes.keys())
        
        for i, signal_class in enumerate(class_names):
            if i >= 16:  # Maximo 16 classes
                break
                
            # Gerar sinal de exemplo
            signal = self.generate_single_signal(signal_class)
            
            # Plotar
            axes[i].plot(self.t * 1000, signal, 'b-', linewidth=1)
            axes[i].set_title(f'{signal_class}', fontsize=10)
            axes[i].set_xlabel('Tempo (ms)')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_plots:
            plot_dir = self.dataset_path / 'plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_dir / 'sample_signals.png', dpi=300, bbox_inches='tight')
            
        plt.show()


def create_power_quality_dataset(force_regenerate: bool = False) -> Dict:
    """
    Funcao principal para criar/carregar dataset de qualidade de energia.
    
    Args:
        force_regenerate: Forcar regeneracao mesmo se dataset existir
        
    Returns:
        Dataset completo
    """
    generator = PowerQualitySignalGenerator()
    
    if generator.dataset_exists() and not force_regenerate:
        print("Dataset ja existe. Carregando...")
        return generator.load_dataset()
    else:
        print("Gerando novo dataset...")
        return generator.generate_dataset(n_samples_per_class=300)


if __name__ == '__main__':
    # Teste do gerador
    generator = PowerQualitySignalGenerator()
    
    # Gerar dataset se nao existir
    if not generator.dataset_exists():
        print("Gerando dataset de sinais de qualidade de energia...")
        dataset = generator.generate_dataset(n_samples_per_class=300)
        print(f"Dataset gerado com {len(dataset['signals'])} sinais")
    else:
        print("Dataset ja existe!")
        dataset = generator.load_dataset()
        print(f"Dataset carregado com {len(dataset['signals'])} sinais")
    
    # Plotar amostras
    print("Gerando graficos de exemplo...")
    generator.plot_sample_signals()