#Beta-NLL loss from 
#@inproceedings{Seitzer2022PitfallsOfUncertainty,
#   title = {On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks},
#   author = {Seitzer, Maximilian and Tavakoli, Arash and Antic, Dimitrije and Martius, Georg},
#   booktitle = {International Conference on Learning Representations},
#   month = apr,
#   year = {2022},
#   url = {https://openreview.net/forum?id=aPOpXlnV1T},
#   month_numeric = {4}
# }
def beta_nll_loss(mean, variance, target, beta=0.5):
    """Compute beta-NLL loss
    
    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative 
        weighting between data points, where `0` corresponds to 
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * (variance.detach() ** beta)
    
    return loss.sum(axis=-1)
