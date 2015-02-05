package fr.inria.tacoma.bft.core.frame.impl;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.ImmutableMassFuntion;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class ImmutableMassFunctionTest {

    private FrameOfDiscernment frame;
    private MassFunction mutable;
    private MassFunction immutable;

    @Before
    public void setUp() {
        frame = FrameOfDiscernment.newFrame("position", "sitting", "standing", "layingDown");
        mutable = new MassFunctionImpl(frame);
        mutable.set(frame.toStateSet("sitting"), 0.3);
        immutable = new ImmutableMassFuntion(mutable);
    }

    @Test(expected = UnsupportedOperationException.class)
    public void ImmutableMassFunction_CallingSet_ThrowsException() {
        immutable.set(frame.toStateSet("sitting"), 0.3);
    }

    @Test(expected = UnsupportedOperationException.class)
    public void ImmutableMassFunction_CallingAddToFocal_ThrowsException() {
        immutable.addToFocal(frame.toStateSet("sitting"), 0.3);
    }

    @Test(expected = UnsupportedOperationException.class)
    public void ImmutableMassFunction_CallingNormalize_ThrowsException() {
        immutable.normalize();
    }

    @Test(expected = UnsupportedOperationException.class)
    public void ImmutableMassFunction_CallingPutRemainingOnIgnorance_ThrowsException() {
        immutable.putRemainingOnIgnorance();
    }

    @Test(expected = UnsupportedOperationException.class)
    public void ImmutableMassFunction_CallingWeaken_ThrowsException() {
        immutable.discount(0.2);
    }

    @Test
    public void ImmutableMassFunction_CallingAnyOtherFunction_ReturnTheSameAsTheUnderlyingFunction() {
        assertEquals(mutable.get("sitting"), immutable.get("sitting"), 1e-12);
        assertEquals(mutable.get(frame.toStateSet("sitting")),
                immutable.get(frame.toStateSet("sitting")), 1e-12);
        assertEquals(mutable.getTotalAssignedMass(), immutable.getTotalAssignedMass(), 1e-12);
        assertEquals(mutable.getFocalStateSets(), immutable.getFocalStateSets());
    }

    @Test
    public void toString_OnUnderlyingFunction_ReturnsTheSameAsImmutable() {
        assertEquals(mutable.toString(),immutable.toString());
    }

    @Test
    public void hashCode_OnUnderlyingFunction_ReturnsTheSameAsImmutable() {
        assertEquals(mutable.hashCode(), immutable.hashCode());
    }
}
