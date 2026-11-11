# McCall job search: model and configuration

An unemployed worker observes a wage offer and chooses whether to accept. Acceptance pays the offered wage immediately and in every subsequent period. Employment is permanent. Rejection pays unemployment benefit $b$ immediately, followed by a new independent wage offer next period. The worker maximizes expected discounted income.

## States, actions, rewards, and transitions

The full state is $(U,w)$ when unemployed with offer $w$, or $(E,w)$ when employed at wage $w$:

$$S=\{(U,w),(E,w):w\in\mathcal W\}.$$

| State | Feasible action | Current reward | Next state |
| --- | --- | --- | --- |
| $(U,w)$ | accept | $w$ | $(E,w)$ with probability 1 |
| $(U,w)$ | reject | $b$ | $(U,w')$ with probability $q(w')$ |
| $(E,w)$ | work | $w$ | $(E,w)$ with probability 1 |

All other transition probabilities are zero. There is no job separation, saving, or search cost beyond the forgone wage.

For any feasible policy, $V^\sigma((E,w))=w/(1-\beta)$. Optimal unemployed values satisfy

$$
V^*((U,w))=\max\left\{\frac{w}{1-\beta},\ b+\beta\sum_{w'\in\mathcal W}q(w')V^*((U,w'))\right\}.
$$

## Baseline configuration

The baseline wage offers, offer probabilities, unemployment benefit, and discount factor are:

```python
wages = (1, 2, 3, 4, 5)
probabilities = (0.1, 0.2, 0.4, 0.2, 0.1)
benefit = 1.0
beta = 0.9
```
