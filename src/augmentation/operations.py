# Code is mainly adopted from https://github.com/moskomule/dda/tree/fasteraa/faster_autoaugment
""" Operations"""

from typing import Optional, Callable, Tuple

import torch
from torch import nn
from torch.distributions import RelaxedBernoulli, Bernoulli

from .functional import eda_word_del, eda_word_swap, eda_word_del_swap, cutoff, cbert, backtrans, r3f, random_mask

__all__ = ['EDA_WordDelete', 'EDA_WordSwap', 'EDA_WordDelete_Swap', 'Cutoff', 'Cbert', 'BackTrans', 'R3F', 'RandomMask']


class _Operation(nn.Module):
    def __init__(self,
                operation: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                initial_magnitude: Optional[float] = None,
                initial_probability: float = 0.5,
                magnitude_range: Optional[Tuple[float, float]] = None,
                probability_range: Optional[Tuple[float, float]] = None,
                temperature: float = 0.1):

        super(_Operation, self).__init__()
        self.operation = operation

        self.magnitude_range = None
        if initial_magnitude is None:
            self._magnitude = None
        elif magnitude_range is None:
            self.register_buffer("_magnitude", torch.empty(1).fill_(initial_magnitude))
        else:
            self._magnitude = nn.Parameter(torch.empty(1).fill_(initial_magnitude))
            assert 0 <= magnitude_range[0] < magnitude_range[1] <= 5
            self.magnitude_range = magnitude_range

        self.probability_range = probability_range
        if self.probability_range is None:
            self.register_buffer("_probability", torch.empty(1).fill_(initial_probability))
        else:
            assert 0 <= initial_probability <= 1
            assert 0 <= self.probability_range[0] < self.probability_range[1] <= 1
            self._probability = nn.Parameter(torch.empty(1).fill_(initial_probability))

        assert 0 < temperature
        self.register_buffer("temperature", torch.empty(1).fill_(temperature))

        # to avoid accessing CUDA tensors in multiprocessing env.
        self._py_magnitude = initial_magnitude
        self._py_probability = initial_probability


    def forward(self,
                args,
                input_ids: torch.Tensor,
                input_emb: torch.Tensor,
                labels: torch.Tensor,
                eda_word_del_aug: torch.Tensor,
                eda_word_swap_aug: torch.Tensor,
                eda_word_del_swap_aug: torch.Tensor,
                bts: torch.Tensor,
                cbt: torch.Tensor,
                tokenizer,
                model: nn.Module) -> torch.Tensor:

        mask = self.get_mask(input_ids.size(0))
        mask = mask.cuda()
        mag = self.magnitude
        # print(f">>>> aug magnitude: {mag}")

        if self.training:
            aug_input, aug_embed = self.operation(
                args, input_ids, input_emb, labels, eda_word_del_aug, eda_word_swap_aug, eda_word_del_swap_aug, bts, cbt, mag, tokenizer, model
                ) ##### detach
            return aug_input, mask * aug_embed + (1 - mask) * input_emb ##### detach
            # aug_input, aug_embed = self.operation(
            #     args, input_ids, input_emb, labels, eda_word_del_aug, eda_word_swap_aug, eda_word_del_swap_aug, bts, cbt, mag.detach(), tokenizer, model
            #     ) ##### detach
            # return aug_input, mask.detach() * aug_embed.detach() + (1 - mask.detach()) * input_emb ##### detach
        else:
            mask.squeeze_()
            num_valid = mask.sum().long()
            if torch.is_tensor(mag):
                if mag.size(0) == 1:
                    mag = mag.repeat(num_valid)
                else:
                    mag = mag[mask == 1]
            if num_valid > 0:
                aug_input, aug_embed = self.operation(input_ids[mask == 1, ...], input_emb[mask == 1, ...], labels[mask == 1],
                                                        indices[mask == 1], mag, model) ##### ...이 대체 뭥미? 가능한 문법인가?
                input_ids[mask == 1, ...] = aug_input
                input_emb[mask == 1, ...] = aug_embed
            return input_ids, input_emb

    def get_mask(self,
                batch_size=None) -> torch.Tensor:
        size = (batch_size, 1)
        if self.training:
            return RelaxedBernoulli(self.temperature, self.probability).rsample(size)
        else:
            return Bernoulli(self.probability).sample(size)

    @property
    def magnitude(self) -> Optional[torch.Tensor]:
        if self._magnitude is None:
            return None
        mag = self._magnitude
        if self.magnitude_range is not None:
            mag = mag.clamp(*self.magnitude_range)
        #m = mag * self.magnitude_scale
        m = mag
        self._py_magnitude = m.item()
        return m

    @property
    def probability(self) -> torch.Tensor:
        if self.probability_range is None:
            return self._probability
        p = self._probability.clamp(*self.probability_range)
        self._py_probability = p.item()
        return p

    def __repr__(self) -> str:
        s = self.__class__.__name__
        if self.probability is not None:
            prob_state = 'frozen' if self.probability_range is None else 'learnable'
            s += f"(probability={self._py_probability:.3f} ({prob_state}), "
        if self.magnitude is not None:
            mag_state = 'frozen' if self.magnitude_range is None else 'learnable'
            s += f"{' ' * len(s)} magnitude={self._py_magnitude:.5f} ({mag_state}), "
        s += f"{' ' * len(s)} temperature={self.temperature.item():.3f})"
        return s


# NLP Operations
# --------------------- Discrete Aug ---------------------
class EDA_WordDelete(_Operation):
    def __init__(self,
                initial_magnitude: float = 0.5,
                initial_probability: float = 0.5,
                magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                probability_range: Optional[Tuple[float, float]] = (0, 1),
                temperature: float = 0.1):
        super(EDA_WordDelete, self).__init__(eda_word_del, initial_magnitude, initial_probability, magnitude_range,
                                            probability_range, temperature)


class EDA_WordSwap(_Operation):
    def __init__(self,
                initial_magnitude: float = 0.5,
                initial_probability: float = 0.5,
                magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                probability_range: Optional[Tuple[float, float]] = (0, 1),
                temperature: float = 0.1):
        super(EDA_WordSwap, self).__init__(eda_word_swap, initial_magnitude, initial_probability, magnitude_range,
                                            probability_range, temperature)


class EDA_WordDelete_Swap(_Operation):
    def __init__(self,
                initial_magnitude: float = 0.5,
                initial_probability: float = 0.5,
                magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                probability_range: Optional[Tuple[float, float]] = (0, 1),
                temperature: float = 0.1):
        super(EDA_WordDelete_Swap, self).__init__(eda_word_del_swap, initial_magnitude, initial_probability, magnitude_range,
                                                probability_range, temperature)


class Cbert(_Operation):
    def __init__(self,
                initial_magnitude: float = 0.5,
                initial_probability: float = 0.5,
                magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                probability_range: Optional[Tuple[float, float]] = (0, 1),
                temperature: float = 0.1):
        super(Cbert, self).__init__(cbert, initial_magnitude, initial_probability, magnitude_range,
                                    probability_range, temperature)

class BackTrans(_Operation):
    def __init__(self,
                initial_magnitude: float = 0.5,
                initial_probability: float = 0.5,
                magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                probability_range: Optional[Tuple[float, float]] = (0, 1),
                temperature: float = 0.1):
        super(BackTrans, self).__init__(backtrans, initial_magnitude, initial_probability, magnitude_range,
                                    probability_range, temperature)

class RandomMask(_Operation):
    def __init__(self,
                    initial_magnitude: float = 0.5,
                    initial_probability: float = 0.5,
                    magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                    probability_range: Optional[Tuple[float, float]] = (0, 1),
                    temperature: float = 0.1):
        super(RandomMask, self).__init__(random_mask, initial_magnitude, initial_probability, magnitude_range,
                                        probability_range, temperature)                                 

# --------------------- Continuous Aug ---------------------
class R3F(_Operation):
    def __init__(self,
                initial_magnitude: float = 1e-5,
                initial_probability: float = 0.5,
                magnitude_range: Optional[Tuple[float, float]] = (1e-6, 1e-1),
                probability_range: Optional[Tuple[float, float]] = (0, 1),
                temperature: float = 0.1):
        super(R3F, self).__init__(r3f, initial_magnitude, initial_probability, magnitude_range,
                                    probability_range, temperature)

class Cutoff(_Operation):
    def __init__(self,
                initial_magnitude: float = 0.3,
                initial_probability: float = 0.5,
                magnitude_range: Optional[Tuple[float, float]] = (0.05, 0.5),
                probability_range: Optional[Tuple[float, float]] = (0, 1),
                temperature: float = 0.1):
        super(Cutoff, self).__init__(cutoff, initial_magnitude, initial_probability, magnitude_range,
                                    probability_range, temperature)

# class Adversarial(_Operation):
#     def __init__(self,
#                  initial_magnitude: float = 0.1,
#                  initial_probability: float = 0.5,
#                  magnitude_range: Optional[Tuple[float, float]] = (0.01, 5.0),
#                  probability_range: Optional[Tuple[float, float]] = (0, 1),
#                  temperature: float = 0.1):
#         super(Adversarial, self).__init__(adversarial, initial_magnitude, initial_probability, magnitude_range,
#                                      probability_range, temperature)