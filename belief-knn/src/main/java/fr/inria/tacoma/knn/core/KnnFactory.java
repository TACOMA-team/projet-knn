package fr.inria.tacoma.knn.core;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.knn.util.KnnUtils;

import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

public abstract class KnnFactory<T> {

    private FrameOfDiscernment frame;
    private Function<List<MassFunction>, MassFunction> combination;
    private BiFunction<T, T, Double> distance;

    public KnnFactory(FrameOfDiscernment frame,
                      Function<List<MassFunction>, MassFunction> combination,
                      BiFunction<T, T, Double> distance) {
        this.frame = frame;
        this.combination = combination;
        this.distance = distance;
    }

    public FrameOfDiscernment getFrame() {
        return frame;
    }

    public void setFrame(FrameOfDiscernment frame) {
        this.frame = frame;
    }

    public Function<List<MassFunction>, MassFunction> getCombination() {
        return combination;
    }

    public void setCombination(
            Function<List<MassFunction>, MassFunction> combination) {
        this.combination = combination;
    }

    public BiFunction<T, T, Double> getDistance() {
        return distance;
    }

    public void setDistance(BiFunction<T, T, Double> distance) {
        this.distance = distance;
    }

    public abstract KnnBelief<T> newKnnBelief(List<? extends LabelledPoint<T>> points,
                                              Map<String, Double> gammaProvider, int k, double alpha);

    public static <T> KnnFactory<T> getGenericFactory(FrameOfDiscernment frame,
                                                      BiFunction<T, T, Double> distance) {

        return new KnnFactory<T>(frame, KnnUtils::optimizedDuboisAndPrade, distance) {
            @Override
            public KnnBelief<T> newKnnBelief(List<? extends LabelledPoint<T>> points,
                                             Map<String, Double> gammaProvider, int k, double alpha) {
                return new GenericKnn<>(points, k, alpha, getFrame(), getCombination(),
                        getDistance(), gammaProvider);
            }
        };

    }

    public static KnnFactory<Double> getDoubleKnnFactory(FrameOfDiscernment frame) {
        return new KnnFactory<Double>(frame, KnnUtils::optimizedDuboisAndPrade,
                (a,b) -> Math.abs(a - b)) {
            @Override
            public KnnBelief<Double> newKnnBelief(List<? extends LabelledPoint<Double>> points,
                                                  Map<String, Double> gammaProvider, int k, double alpha) {
                return new DoubleKnnBelief((List<LabelledPoint<Double>>) points, k,
                        alpha, getFrame(), getCombination(), getDistance(), gammaProvider);
            }
        };
    }

    public static KnnFactory<Double> getDoubleDempsterFactory(FrameOfDiscernment frame) {
        return new KnnFactory<Double>(frame, KnnUtils::optimizedDuboisAndPrade,
                (a,b) -> Math.abs(a - b)) {
            @Override
            public KnnBelief<Double> newKnnBelief(List<? extends LabelledPoint<Double>> points,
                                                  Map<String, Double> gammaProvider, int k, double alpha) {
                return new DempsterDoubleKnn((List<LabelledPoint<Double>>) points, k,
                        alpha, getFrame(), getDistance(), gammaProvider);
            }
        };
    }
}
