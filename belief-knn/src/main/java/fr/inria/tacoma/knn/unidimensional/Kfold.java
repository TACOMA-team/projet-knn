package fr.inria.tacoma.knn.unidimensional;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.core.LabelledPoint;
import fr.inria.tacoma.knn.util.AveragingBeliefModel;
import fr.inria.tacoma.knn.util.KnnUtils;

import java.util.*;
import java.util.function.BiFunction;

public class Kfold<T> {

    private final BiFunction<T, T, Double> distance;
    private final List<LabelledPoint<T>> samples;
    private final int k;
    private final Random random;
    private final FrameOfDiscernment frame;


    public Kfold(FrameOfDiscernment frame, List<LabelledPoint<T>> samples, int k,
                 BiFunction<T, T, Double> distance) {
        this.samples = samples;
        this.k = k;
        this.distance = distance;
        this.frame = frame;
        this.random = new Random();
    }

    public SensorBeliefModel<T> generateModel() {
        List<LabelledPoint<T>> shuffled = new ArrayList<>(samples);
        Collections.shuffle(shuffled);
        List<List<LabelledPoint<T>>> sublists = KnnUtils.split(shuffled, k);
        List<SensorBeliefModel<T>> models = new ArrayList<>(k);

        for (int validationIndex = 0; validationIndex < k; validationIndex++) {
            List<LabelledPoint<T>> trainingSet = new ArrayList<>();
            List<LabelledPoint<T>> crossValidation = sublists.get(validationIndex);
            for (int j = 0; j < k; j++) {
                if(validationIndex != j) {
                    trainingSet.addAll(sublists.get(j));
                }
            }
            KnnBelief<T> bestModel = KnnUtils
                    .getBestKnnBeliefForAlphaAndK(frame, trainingSet, crossValidation,
                    distance);
            models.add(bestModel);
        }

        return new AveragingBeliefModel<>(models);
    }


}
