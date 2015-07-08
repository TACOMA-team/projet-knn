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
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
//        List<LabelledPoint<T>> shuffled = new ArrayList<>(samples);
//        Collections.shuffle(shuffled, random);
//        List<List<LabelledPoint<T>>> sublists = KnnUtils.split(shuffled, k);
        List<List<LabelledPoint<T>>> sublists = createSubLists(samples, k);

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

    private <T> List<List<T>> createSubLists(List<T> list, int nb) {
        List<List<T>> sublists = new ArrayList<>();
        for (int i = 0; i < nb; i++) {
            sublists.add(new ArrayList<>());
        }

        int sublistIndex = 0;
        for (T object : list) {
            sublists.get(sublistIndex).add(object);
            sublistIndex++;
            if(sublistIndex >= sublists.size()) {
                sublistIndex = 0;
            }
        }
        return sublists;
    }

}
