package fr.inria.tacoma.knn;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.bft.util.Mass;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class KnnBelief implements SensorBeliefModel<Double> {

    private final int k;
    private final double alpha;
    private final FrameOfDiscernment frame;
    private final List<Point> points;
    private final Map<String, Double> gammaProvider;
    Function<List<MassFunction>, MassFunction> combination;

    public KnnBelief(TrainingSet trainingSet, int k, double alpha,
                     FrameOfDiscernment frame,
                     Function<List<MassFunction>, MassFunction> combination) {
        this.k = k;
        this.alpha = alpha;
        this.frame = frame;
        this.combination = combination;
        this.points = trainingSet.getPoints();
        this.gammaProvider = trainingSet.getStandardDevs();
    }

    private List<Point> knn(double value) {
        return points.stream()
                .sorted((p1,p2) ->
                        Double.compare(Math.abs(p1.getValue() - value),Math.abs(p2.getValue() - value)))
                .limit(k).collect(Collectors.toList());
    }

    private MassFunction getMassFunction(double value, Point point) {
        MassFunction mass = new MassFunctionImpl(frame);
        double gamma = 1.0 / gammaProvider.get(point.getLabel());
        mass.addToFocal(frame.toStateSet(point.getLabel()),
                alpha * Math.exp(- Math.pow(Math.abs(value - point.getValue()) * gamma, 1)));
        mass.putRemainingOnIgnorance();
        return mass;
    }

    public int getK() {
        return k;
    }

    public double getAlpha() {
        return alpha;
    }

    @Override
    public MassFunction toMass(Double sensorValue) {
        List<Point> knn = knn(sensorValue);
        List<MassFunction> masses = knn.stream()
                .map(p -> getMassFunction(sensorValue, p))
                .collect(Collectors.toList());
        return Mass.toConsonant(combination.apply(masses));
    }


    @Override
    public MassFunction toMassWithoutValue() {
        MassFunctionImpl massFunction = new MassFunctionImpl(frame);
        massFunction.putRemainingOnIgnorance();
        return massFunction;
    }


    @Override
    public FrameOfDiscernment getFrame() {
        return frame;
    }


}
