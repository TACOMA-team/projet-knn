package fr.inria.tacoma.knn.util;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.bft.util.Mass;

public class ConsonantBeliefModel<T> implements SensorBeliefModel<T> {

    SensorBeliefModel<T> underlyingModel;

    public ConsonantBeliefModel(SensorBeliefModel<T> underlyingModel) {
        this.underlyingModel = underlyingModel;
    }

    @Override
    public MassFunction toMass(T sensorValue) {
        return Mass.toConsonant(underlyingModel.toMass(sensorValue));
    }

    @Override
    public MassFunction toMassWithoutValue() {
        return Mass.toConsonant(underlyingModel.toMassWithoutValue());
    }

    @Override
    public FrameOfDiscernment getFrame() {
        return underlyingModel.getFrame();
    }
}
