# FedDAA: Dynamic Client Clustering for Concept Drift Adaptation in Federated Learning



## Environments

```
pillow=11.0.0
tqdm=4.66.5
scikit-learn=1.5.2
numpy=1.26.4
pytorch=1.12.1
matplotlib=3.9.2
networkx=3.2.1
cvxpy=1.6.0
torchvision=0.13.1
cuda-version=10.2
cudatoolkit=10.2.89
```

## Data

Please split data by the following scripts. 


- CIFAR-10
```
python create_c/make_cifar_c-60_client-simple2-iid-4concept-change-name-version2.py
```

- CIFAR-100
```
python create_c/make_cifar_100_c-60_client-simple2-iid-4concept-change-name-version2.py
```

- Fashion-MNIST
```
python create_c/make_fmnist_c-60_client-simple2-iid-4concept-change-name-version2.py
```

## Commands
- CIFAR-10
```
python FedDAA_CIFAR10.py cifar10-c fedrc_store_history --n_learners 2 --bz 128 --lr 0.06 --lr_scheduler constant --log_freq 1 --optimizer sgd --seed 1 --verbose 1 --T 6 --n_rounds 40 --device 0 --sampling_rate 0.5 --suffix T_6-client_60-FedDAA-CIFAR-10
```
- CIFAR-100
```
python FedDAA_CIFAR100.py cifar100-c fedrc_store_history --n_learners 2 --bz 128 --lr 0.06 --lr_scheduler constant --log_freq 1 --optimizer sgd --seed 1 --verbose 1 --T 6 --n_rounds 40 --device 0 --sampling_rate 0.5 --suffix T_6--client_60-FedDAA-CIFAR-100
```
- Fashion-MNIST
```
python FedDAA_Fashion_MNIST.py fmnist-c fedrc_store_history --n_learners 2 --bz 128 --lr 0.06 --lr_scheduler constant --log_freq 1 --optimizer sgd --seed 1 --verbose 1 --T 6 --n_rounds 40 --device 0 --sampling_rate 0.5 --suffix T_6-client_60-FedDAA-Fashion-MNIST


```





