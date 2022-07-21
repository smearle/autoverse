import qdax
from qdax.core.map_elites import MAPElites

# Instantiate MAP-Elites
map_elites = MAPElites(
    scoring_function=scoring_fn,
    emitter=mixing_emitter,
    metrics_function=metrics_function,
)

# Initializes repertoire and emitter state
repertoire, emitter_state, random_key = map_elites.init(init_variables, centroids, random_key)

# Run MAP-Elites loop
for i in range(num_iterations):
    (repertoire, emitter_state, metrics, random_key,) = map_elites.update(
            repertoire,
            emitter_state,
            random_key,
        )

# Get contents of repertoire
repertoire.genotypes, repertoire.fitnesses, repertoire.descriptors