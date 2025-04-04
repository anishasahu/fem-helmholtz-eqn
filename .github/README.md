```
uv sync
pre-commit install
```

Run using 

```
uv run runner.py
```


|           class            | inner boudary | outer boundary |
| :------------------------: | :-----------: | :------------: |
|`fem2d_dirichlet`           |   dirichlet   |   dirichlet    |
|`fem2d_dirichlet_sommerfeld`|   dirichlet   |   sommerfeld   |
|`fem2d_neumann_dirichlet`   |    neumann    |   dirichlet    |
|`fem2d`                     |    neumann    |   sommerfeld   |
