package fr.inria.tacoma.knn.util;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MutableMass;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.bft.util.Mass;

public class ConsonantBeliefModel<T> implements SensorBeliefModel<T> {

    SensorBeliefModel<T> underlyingModel;

    public ConsonantBeliefModel(SensorBeliefModel<T> underlyingModel) {
        this.underlyingModel = underlyingModel;
    }

    @Override
    public MutableMass toMass(T sensorValue) {
        //FIXME this is quick and dirty. Should work with current version of jbft
        return (MutableMass)Mass.toConsonant(underlyingModel.toMass(sensorValue));
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
