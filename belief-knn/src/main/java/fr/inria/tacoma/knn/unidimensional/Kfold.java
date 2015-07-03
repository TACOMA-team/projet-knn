package fr.inria.tacoma.knn.unidimensional;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.core.KnnFactory;
import fr.inria.tacoma.knn.core.LabelledPoint;
import fr.inria.tacoma.knn.util.AveragingBeliefModel;
import fr.inria.tacoma.knn.util.KnnUtils;

import java.util.*;
import java.util.function.BiFunction;

public class Kfold<T> {

    private final List<LabelledPoint<T>> samples;
    private final int k;
    private final Random random;
    private final KnnFactory<T> factory;


    public Kfold(KnnFactory<T> factory, List<LabelledPoint<T>> samples, int k) {
        this.samples = samples;
        this.k = k;
        this.factory = factory;
        this.random = new Random();
    }

    public SensorBeliefModel<T> generateModel() {
        List<LabelledPoint<T>> shuffled = new ArrayList<>(samples);
        Collections.shuffle(shuffled, random);
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
            KnnBelief<T> bestModel =
                    KnnUtils.getBestKnnBeliefForAlphaAndK(factory, trainingSet, crossValidation);
            models.add(bestModel);
        }

        return new AveragingBeliefModel<>(models);
    }


}
