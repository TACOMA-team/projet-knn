package fr.inria.tacoma.knn.util;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MutableMass;
import fr.inria.tacoma.bft.criteria.Criteria;
import fr.inria.tacoma.bft.decision.CriteriaDecisionStrategy;
import fr.inria.tacoma.bft.decision.Decision;
import fr.inria.tacoma.bft.decision.DecisionStrategy;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.knn.core.LabelledPoint;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class DiscountingBeliefModel implements SensorBeliefModel<Double> {

    private SensorBeliefModel<Double> underlyingModel;
    private Function<Double, Double> weakeningFunction;

    public DiscountingBeliefModel(SensorBeliefModel<Double> underlyingModel,
                                  Function<Double, Double> weakeningFunction) {
        this.underlyingModel = underlyingModel;
        this.weakeningFunction = weakeningFunction;
    }

    @Override
    public MutableMass toMass(Double sensorValue) {
        MutableMass massFunction = underlyingModel.toMass(sensorValue);
        massFunction.discount(weakeningFunction.apply(sensorValue));
        return massFunction;
    }

    @Override
    public MassFunction toMassWithoutValue() {
        return underlyingModel.toMassWithoutValue();
    }

    @Override
    public FrameOfDiscernment getFrame() {
        return underlyingModel.getFrame();
    }

    public static class WeakeningFunction implements Function<Double, Double> {
        private Map<Double, Double> alphas = new HashMap<>();
        private Map<Double, Double> stdDevs = new HashMap<>();

        public void putWeakeningPoint(double center, double weakening, double amplitude) {
            alphas.put(center, weakening);
            stdDevs.put(center, amplitude);
        }

        public double getWeakening(double value) {
            return alphas.getOrDefault(value, 0.0);
        }

        @Override
        public Double apply(Double value) {
            return alphas.entrySet().stream().mapToDouble(entry -> {
                Double max = entry.getValue();
                double diff = entry.getKey() - value;
                return max * Math.exp(-Math.pow(3 * diff / stdDevs.get(entry.getKey()), 2));
            }).sum();
        }
    }

    public static DiscountingBeliefModel
    generateDiscountingModel(SensorBeliefModel<Double> model,
                             List<LabelledPoint<Double>> crossValidation,
                             DecisionStrategy decisionStrategy,
                             Map<String, Double> weakeningRanges) {
        DiscountingBeliefModel.WeakeningFunction weakeningFunction = new DiscountingBeliefModel.WeakeningFunction();
        DiscountingBeliefModel discountingBeliefModel =
                new DiscountingBeliefModel(model, weakeningFunction);

        boolean wrongDecisionOccured = true;
        while (wrongDecisionOccured) {
            wrongDecisionOccured = false;
            for (LabelledPoint<Double> measure : crossValidation) {
                StateSet expectedDecision = model.getFrame().toStateSet(measure.getLabel());
                StateSet actualDecision;

                Decision decision = decisionStrategy.decide(
                        discountingBeliefModel.toMass(measure.getValue()));
                actualDecision = decision.getStateSet();

                if (!actualDecision.includesOrEquals(expectedDecision)) {

                    System.out.println("bad decision for value: " + measure.getValue());
                    System.out.println(
                            "actualDecision=" + actualDecision + " real = " + expectedDecision);
                    System.out.println(decision);

                    double weakening = 0.1 + weakeningFunction.getWeakening(measure.getValue());
                    System.out.println("applying discounting " + weakening);

                    weakeningFunction.putWeakeningPoint(measure.getValue(), weakening,
                            weakeningRanges.get(measure.getLabel()));
                    wrongDecisionOccured = true;
                }
            }

        }

        return discountingBeliefModel;
    }
}
