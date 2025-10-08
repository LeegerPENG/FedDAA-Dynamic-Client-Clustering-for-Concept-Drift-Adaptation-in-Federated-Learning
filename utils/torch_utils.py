import warnings

import torch
import torch.nn as nn
import copy


def global_steps(
        learners,
        target_learner,
        weights=None,
        global_lr=1.0):

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device='cpu')

    else:
        weights = weights.to('cpu')
    
    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            data_o = target_state_dict[key].data.clone()

            target_state_dict[key].data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                target_state_dict[key].data += weights[learner_id] * (data_o - state_dict[key].data.clone())
            
            target_state_dict[key].data = data_o - global_lr * target_state_dict[key].data

        else:
            # tracked batches
            data_o = target_state_dict[key].data.clone()
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()



def average_learners(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False):
    """
    Compute the average of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device='cpu')

    else:
        weights = weights.to('cpu')

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()

                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()

def average_learners_for_FLASH(
        learners,
        target_learner,
        m,
        v,
        d,
        flag):



    ################################
    beta_1 = 0.5
    beta_2 = 0.5
    yita_global = 1
    ###########################

    tau = 1e-9
    ##########################################

    w_locals = [learner.model.state_dict(keep_vars=True) for learner in learners]
    w_global = target_learner.model.state_dict(keep_vars=True)


    delta = copy.deepcopy(w_locals[0]) 
    for k in delta.keys():
        delta[k] = torch.zeros_like(delta[k], dtype=torch.float)  
        for i in range(1, len(w_locals)):
            # print(type(w_locals[i][k]), type(w_global[k]))
            delta[k] += (w_locals[i][k] - w_global[k])
        delta[k] = torch.div(delta[k], len(w_locals))
        # DEBUG
        # print('delta[k]:{}'.format(delta[k]))


    if flag == 0:

        m_new = {k: m[k].clone() for k in m.keys()}
        for k in m_new.keys():
            m_new[k] = delta[k]
    else:
        m_new = {k: m[k].clone() for k in m.keys()}
        for k in m_new.keys():
            m_new[k] = beta_1 * m[k] + (1 - beta_1) * delta[k]
            # DEBUG
            # print('m_new[k]:{}'.format(m_new[k]))


    if flag == 0:

        v_new = {k: v[k].clone() for k in v.keys()}
        for k in v_new.keys():
            v_new[k] = torch.pow(delta[k], 2)
    else:
        v_new = {k: v[k].clone() for k in v.keys()}
        for k in v_new.keys():
            v_new[k] = beta_2 * v[k] + (1 - beta_2) * torch.pow(delta[k], 2)
            # DEBUG
            # print('v_new[k]:{}'.format(v_new[k]))



    if flag == 0:
        # DEBUG
        v_mold = {k: v_new[k].clone() for k in v_new.keys()}
        # v_mold = v_new.data.clone()
        for k in v_mold.keys():
            v_mold[k] = torch.norm(v_mold[k])
            # DEBUG
            # print('v_mold[k]:{}'.format(v_mold[k]))
    else:

        v_mold = {k: v[k].clone() for k in v.keys()}
        for k in v_mold.keys():
            v_mold[k] = torch.norm(v_mold[k])
            # DEBUG
            # print('v_mold[k]:{}'.format(v_mold[k]))


    #delta_v_difference_mold = copy.deepcopy(delta)
    delta_v_difference_mold = {k: torch.zeros_like(v_new[k], dtype=torch.float) for k in v_new.keys()}
    for k in delta.keys():
        delta_v_difference_mold[k] = torch.norm(torch.pow(delta[k], 2) - v_new[k])
        # DEBUG
        # print('delta_v_difference_mold[k]:{}'.format(delta_v_difference_mold[k]))


    # beta_3 = copy.deepcopy(v_mold)
    beta_3 = {k: torch.zeros_like(v_mold[k], dtype=torch.float) for k in v_mold.keys()}
    for k in beta_3.keys():
        beta_3[k] = torch.div(v_mold[k], torch.add(delta_v_difference_mold[k], v_mold[k]))
        # DEBUG
        # print('beta_3[k]:{}'.format(beta_3[k])) 

    if flag == 0:

        d_new = {k: d[k].clone() for k in d.keys()}
        for k in d_new.keys():
            d_new[k] = torch.pow(delta[k], 2) - v_new[k]
    else:

        d_new = {k: torch.zeros_like(d[k], dtype=torch.float) for k in d.keys()}
        for k in d_new.keys():
            d_new[k] = beta_3[k] * d[k] + (1 - beta_3[k]) * (torch.pow(delta[k], 2) - v_new[k])


    for k in w_global.keys():

        if 'bn' in k:
            # new_w_global[k] = w_global[k] + yita_global * m_new[k]
            w_global[k] = w_global[k] + m_new[k]

        else:
            # new_w_global[k] = w_global[k] + yita_global * m_new[k]
            w_global[k] = w_global[k] + yita_global * torch.div(m_new[k],
                                                                    torch.sqrt(v_new[k]) - d_new[k] + tau)
        # if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k :
        #     new_w_global[k] = w_global[k] + yita_global *torch.div( m_new[k],torch.sqrt(v_new[k])-d_new[k]+tau)
        # new_w_global[k] = w_global[k] + yita_global * m_new[k]
        # DEBUG
        # print(k)
        # print('new_w_global[k]:{}'.format(new_w_global[k]))
    

    target_learner.model.load_state_dict(w_global)
    return m_new,v_new,d_new




def partial_average(learners, average_learner, alpha):
    """
    performs a step towards aggregation for learners, i.e.

    .. math::
        \forall i,~x_{i}^{k+1} = (1-\alpha) x_{i}^{k} + \alpha \bar{x}^{k}

    :param learners:
    :type learners: List[Learner]
    :param average_learner:
    :type average_learner: Learner
    :param alpha:  expected to be in the range [0, 1]
    :type: float

    """
    source_state_dict = average_learner.model.state_dict()

    target_state_dicts = [learner.model.state_dict() for learner in learners]

    for key in source_state_dict:
        if source_state_dict[key].data.dtype == torch.float32:
            for target_state_dict in target_state_dicts:
                target_state_dict[key].data =\
                    (1-alpha) * target_state_dict[key].data + alpha * source_state_dict[key].data


def differentiate_learner(target, reference_state_dict, coeff=1.):
    """
    set the gradient of the model to be the difference between `target` and `reference` multiplied by `coeff`

    :param target:
    :type target: Learner
    :param reference_state_dict:
    :type reference_state_dict: OrderedDict[str, Tensor]
    :param coeff: default is 1.
    :type: float

    """
    target_state_dict = target.model.state_dict(keep_vars=True)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:

            target_state_dict[key].grad = \
                coeff * (target_state_dict[key].data.clone() - reference_state_dict[key].data.clone())


def copy_model(target, source):
    """
    Copy learners_weights from target to source
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None

    """
    target.load_state_dict(source.state_dict())


def get_learner_distance(learner_1, learner_2):
    net1_state_dict = learner_1.model.state_dict()
    net2_state_dict = learner_2.model.state_dict()
    distance = 0
    for key in net1_state_dict:
        distance += torch.norm((net1_state_dict[key] - net2_state_dict[key]).float())
    return distance.item()


def simplex_projection(v, s=1):
    """
    Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):

    .. math::
        min_w 0.5 * || w - v ||_2^2,~s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) torch tensor,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) torch tensor,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.

    References
    ----------
    [1] Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection
        onto the probability simplex: An efficient algorithm with a
        simple proof, and an application." arXiv preprint
        arXiv:1309.1541 (2013)
        https://arxiv.org/pdf/1309.1541.pdf

    """

    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape

    u, _ = torch.sort(v, descending=True)

    cssv = torch.cumsum(u, dim=0)

    rho = int(torch.nonzero(u * torch.arange(1, n + 1) > (cssv - s))[-1][0])

    lambda_ = - float(cssv[rho] - s) / (1 + rho)

    w = v + lambda_

    w = (w * (w > 0)).clip(min=0)

    return w


