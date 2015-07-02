package fr.inria.tacoma;

import fr.inria.tacoma.knn.util.KnnUtils;
import org.hamcrest.core.AllOf;
import org.junit.Assert;
import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.hamcrest.MatcherAssert.assertThat;


public class KnnTest {

    @Test
    public void testSplit() {
        List<Integer> integers = IntStream.range(0, 500).boxed().collect(Collectors.toList());

        List<List<Integer>> split = KnnUtils.split(integers, 3);
        Assert.assertTrue(split.stream().allMatch(list -> integers.containsAll(list)));
        Assert.assertFalse(integers.isEmpty());
        split.stream().forEach(list -> integers.removeAll(list));
        Assert.assertTrue(integers.isEmpty());
    }
}
