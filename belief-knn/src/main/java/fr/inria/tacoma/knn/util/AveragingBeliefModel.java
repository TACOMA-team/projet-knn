package fr.inria.tacoma.knn.util;

import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MutableMass;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;

import java.util.List;
import java.util.stream.Collectors;

public class AveragingBeliefModel<T> implements SensorBeliefModel<T>{

    private final FrameOfDiscernment frame;
    List<SensorBeliefModel<T>> beliefModels;

    public AveragingBeliefModel(List<SensorBeliefModel<T>> beliefModels) {
        this.beliefModels = beliefModels;
        this.frame = beliefModels.get(0).getFrame();
    }

    @Override
    public MutableMass toMass(T data) {
        List<MutableMass> masses = beliefModels.stream().map(model -> model.toMass(data))
                .collect(Collectors.toList());
        return Combinations.average(masses);
    }

    @Override
    public MassFunction toMassWithoutValue() {
        List<MassFunction> masses = beliefModels.stream().map(
                SensorBeliefModel::toMassWithoutValue)
                .collect(Collectors.toList());
        return Combinations.average(masses);
    }

    @Override
    public FrameOfDiscernment getFrame() {
        return frame;
    }

    public List<SensorBeliefModel<T>> getBeliefModels() {
        return beliefModels;
    }
}
