package fr.inria.tacoma.bft.core.mass;

import fr.inria.tacoma.bft.core.frame.StateSet;

public interface FocalElementConsumer {
    void apply(StateSet set, double value);
}
