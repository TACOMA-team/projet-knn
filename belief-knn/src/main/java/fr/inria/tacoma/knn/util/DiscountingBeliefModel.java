package fr.inria.tacoma.knn.util;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MutableMass;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;

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
}
