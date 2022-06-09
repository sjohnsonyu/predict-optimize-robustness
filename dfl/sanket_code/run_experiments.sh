RANK=2
INSTANCES=20
PROBLEM="bipartitematching"
NUM_FAKE_FEATURES=0
SAMPLING="random"
TESTINSTANCES=7
SAMPLINGSTD=0.1
NODES=50
ITERS=500
for LOSS in ce
do
    for NUM_SAMPLES in 5000
    do
        for SEED in {1..10}
        do
            echo "Seed: ${SEED}, Loss: ${LOSS}, Samples: ${NUM_SAMPLES}, Sampling: ${SAMPLING}, Rank: ${RANK}, Fake Features: ${NUM_FAKE_FEATURES}, Instances: ${INSTANCES}"
            python3 main.py --instances=$INSTANCES --iters=$ITERS --nodes=$NODES --testinstances=$TESTINSTANCES --problem=$PROBLEM --loss=$LOSS --sampling=$SAMPLING --samplingstd=$SAMPLINGSTD --numsamples=$NUM_SAMPLES --quadrank=$RANK --fakefeatures=$NUM_FAKE_FEATURES --seed=$SEED >> results/${PROBLEM}_${INSTANCES}instances_${NUM_FAKE_FEATURES}noise_${LOSS}_${SAMPLING}${SAMPLINGSTD}_${ITERS}_${NUM_SAMPLES}samples_${RANK}rank_${SEED}.txt
        done
    done
done