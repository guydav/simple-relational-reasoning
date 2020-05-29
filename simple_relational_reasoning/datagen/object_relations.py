from abc import abstractmethod
import random
import torch


class ObjectRelation:
    def __init__(self, field_slices, field_generators, position_field_names=('x', 'y')):
        """
        :param field_slices: A dict from str -> slice, where each field is in the vector objects
        :param field_generators: A dict from str -> field generator object, useful if you need access to field
        properties in order to do the balancing
        """
        self.field_slices = field_slices
        self.field_generators = field_generators

        self.position_field_names = position_field_names
        self.position_field_slices = [self.field_slices[name] for name in self.position_field_names]
        self.position_field_generators = [self.field_generators[name] for name in self.position_field_names]

    @abstractmethod
    def evaluate(self, objects):
        """
        Evaluate the relation on a set of objects, returning the label we expect the model to predict for
        this relation on this set of objects
        :param objects: The set of objects to evaluate over
        :return: The label we expect the model to predict
        """
        pass

    @abstractmethod
    def balance(self, objects, current_label):
        """
        Balance a set of objects, changing the label in a particular direction, in order to enable creating a
        dataset that's more balanced
        :param objects: The set of objects to balance over
        :return: The set of objects, with the opposite label
        """
        pass

    def _object_in_position(self, objects, position):
        object_positions = torch.cat([objects[:, field_slice] for field_slice in self.position_field_slices],
                                     dim=1).to(torch.float).unsqueeze(0)

        matching_indices = torch.nonzero((object_positions == position).all(dim=1)).squeeze()
        return len(matching_indices) > 0, matching_indices


class OneDAdjacentRelation(ObjectRelation):
    def __init__(self, field_slices, field_generators, field_name='x', position_field_names=('x', 'y')):
        super(OneDAdjacentRelation, self).__init__(field_slices, field_generators, position_field_names)
        self.relevant_field_name = field_name
        self.relevant_field_slice = self.field_slices[self.relevant_field_name]
        self.relevant_field_generator = self.field_generators[self.relevant_field_name]

    def evaluate(self, objects):
        positions = objects[:, self.relevant_field_slice]
        l1_distances = torch.cdist(positions, positions, 1)
        return torch.any(torch.isclose(l1_distances, torch.tensor([1.0])))

    def balance(self, objects, current_label):
        if current_label != 0:
            raise ValueError('Can only balance negative cases to positive ones for the time being')

        index_to_modify, index_to_set_next_to = torch.randperm(objects.shape[0])[:2]
        # set x and y to be the same, then modify one
        objects[index_to_modify, self.relevant_field_slice] = objects[index_to_set_next_to, self.relevant_field_slice]

        # if at the edge of the grid, shift in the only valid direction
        if objects[index_to_modify, self.relevant_field_slice] == self.relevant_field_generator.min_coord:
            objects[index_to_modify, self.relevant_field_slice] = self.relevant_field_generator.min_coord + 1

        elif objects[index_to_modify, self.relevant_field_slice] == self.relevant_field_generator.max_coord - 1:
            objects[index_to_modify, self.relevant_field_slice] = self.relevant_field_generator.max_coord - 2

        # if not, shift in either direction
        else:
            objects[index_to_modify, self.relevant_field_slice] += torch.sign(torch.rand(tuple()) - 0.5)

        return objects


class MultipleDAdjacentRelation(ObjectRelation):
    def __init__(self, field_slices, field_generators, position_field_names=('x', 'y')):
        super(MultipleDAdjacentRelation, self).__init__(field_slices, field_generators, position_field_names)

    def evaluate(self, objects):
        positions = torch.cat([objects[:, field_slice] for field_slice in self.position_field_slices],
                              dim=1).to(torch.float).unsqueeze(0)
        l1_distances = torch.cdist(positions, positions, 1)
        return torch.any(torch.isclose(l1_distances, torch.tensor([1.0])))

    def balance(self, objects, current_label):
        if current_label != 0:
            raise ValueError('Can only balance negative cases to positive ones for the time being')

        index_to_modify, index_to_set_next_to = torch.randperm(objects.shape[0])[:2]
        # set x and y to be the same, then modify one
        for field_slice in self.position_field_slices:
            objects[index_to_modify, field_slice] = objects[index_to_set_next_to, field_slice]

        slice_index = random.randint(0, len(self.position_field_names) - 1)
        slice_to_modify = self.position_field_slices[slice_index]
        generator_to_modify = self.position_field_generators[slice_index]

        # if at the edge of the grid, shift in the only valid direction
        if objects[index_to_modify, slice_to_modify] == generator_to_modify.min_coord:
            objects[index_to_modify, slice_to_modify] = generator_to_modify.min_coord + 1

        elif objects[index_to_modify, slice_to_modify] == generator_to_modify.max_coord - 1:
            objects[index_to_modify, slice_to_modify] = generator_to_modify.max_coord - 2

        # if not, shift in either direction
        else:
            objects[index_to_modify, slice_to_modify] += torch.sign(torch.rand(tuple()) - 0.5)

        return objects


class ColorAboveColorRelation(ObjectRelation):
    def __init__(self, field_slices, field_generators, color_field_name='color',
                 above_color_index=0, below_color_index=1, dtype=torch.float, position_field_names=('x', 'y'),
                 stuck_count_to_perturb_x=10):
        super(ColorAboveColorRelation, self).__init__(field_slices, field_generators, position_field_names)
        self.x_field_name = position_field_names[0]
        self.x_field_slice = self.field_slices[self.x_field_name]
        self.x_field_gen = self.field_generators[self.x_field_name]

        self.y_field_name = position_field_names[1]
        self.y_field_slice = self.field_slices[self.y_field_name]
        self.y_field_gen = self.field_generators[self.y_field_name]

        self.dtype = dtype

        self.color_field_name = color_field_name
        self.color_field_slice = self.field_slices[self.color_field_name]
        self.color_field_gen = self.field_generators[self.color_field_name]

        self.above_color_index = above_color_index
        self.above_color_tensor = torch.zeros(self.color_field_gen.n_types, dtype=self.dtype)
        self.above_color_tensor[self.above_color_index] = 1

        self.below_color_index = below_color_index
        self.below_color_tensor = torch.zeros(self.color_field_gen.n_types, dtype=self.dtype)
        self.below_color_tensor[self.below_color_index] = 1

        self.stuck_count_to_perturb_x = stuck_count_to_perturb_x

    def evaluate(self, objects):
        colors = objects[:, self.color_field_slice]
        above_color_y_positions = objects[colors.eq(self.above_color_tensor).all(dim=1), self.y_field_slice]
        below_color_y_positions = objects[colors.eq(self.below_color_tensor).all(dim=1), self.y_field_slice]

        if len(above_color_y_positions) == 0:
            return torch.tensor(False)

        if len(below_color_y_positions) == 0:
            return torch.tensor(True)

        return (above_color_y_positions.view(-1, 1) >= below_color_y_positions.view(1, -1)).all(dim=1).any()

    def balance(self, objects, current_label):
        if current_label != 0:
            raise ValueError('Can only balance negative cases to positive ones for the time being')

        colors = objects[:, self.color_field_slice]
        above_color_y_positions = objects[colors.eq(self.above_color_tensor).all(dim=1), self.y_field_slice]

        # If no objects in the above color exist, create one
        if len(above_color_y_positions) == 0:
            index_to_modify = random.randint(0, objects.shape[0] - 1)
            objects[index_to_modify, self.color_field_slice] = self.above_color_tensor
            colors[index_to_modify, :] = self.above_color_tensor

        below_color_y_positions = objects[colors.eq(self.below_color_tensor).all(dim=1), self.y_field_slice]

        if below_color_y_positions.shape[0] == 0:
            max_below_color_position = self.y_field_gen.min_coord
        else:
            max_below_color_position = int(below_color_y_positions.max())

        above_object_indices = torch.nonzero(colors.eq(self.above_color_tensor).all(dim=1)).squeeze()
        if len(above_object_indices.shape) == 0:
            index_to_modify = int(above_object_indices)
        else:
            index_to_modify = above_object_indices[torch.randperm(above_object_indices.shape[0])[0]]

        collision = True
        stuck_count = 0
        while collision:
            stuck_count += 1

            new_above_color_x_position = objects[index_to_modify, self.x_field_slice]
            if stuck_count > self.stuck_count_to_perturb_x:
                new_above_color_x_position = random.randint(self.x_field_gen.min_coord, self.x_field_gen.max_coord - 1)

            new_above_color_y_position = random.randint(max_below_color_position, self.y_field_gen.max_coord - 1)
            tentative_new_position = torch.tensor([new_above_color_x_position, new_above_color_y_position])
            collision, _ = self._object_in_position(objects, tentative_new_position)

        objects[index_to_modify, self.x_field_slice] = new_above_color_x_position
        objects[index_to_modify, self.y_field_slice] = new_above_color_y_position
        return objects


class ObjectCountRelation(ObjectRelation):
    def __init__(self, field_slices, field_generators, first_field_name='color', first_field_index=0,
                 second_field_name='shape', second_field_index=0, dtype=torch.float, position_field_names=('x', 'y')):
        super(ObjectCountRelation, self).__init__(field_slices, field_generators, position_field_names)
        self.dtype = dtype

        self.first_field_name = first_field_name
        self.first_field_slice = self.field_slices[self.first_field_name]
        self.first_field_gen = self.field_generators[self.first_field_name]
        self.first_object_tensor = torch.zeros(self.first_field_gen.n_types, dtype=self.dtype)
        self.first_field_index = first_field_index
        self.first_object_tensor[self.first_field_index] = 1

        self.second_field_name = second_field_name
        self.second_field_slice = self.field_slices[self.second_field_name]
        self.second_field_gen = self.field_generators[self.second_field_name]
        self.second_object_tensor = torch.zeros(self.second_field_gen.n_types, dtype=self.dtype)
        self.second_field_index = second_field_index
        self.second_object_tensor[self.second_field_index] = 1

    def evaluate(self, objects):
        first_objects = objects[:, self.first_field_slice]
        second_objects = objects[:, self.second_field_slice]
        first_object_count = first_objects.eq(self.first_object_tensor).all(dim=1).sum()
        second_object_count = second_objects.eq(self.second_object_tensor).all(dim=1).sum()
        return first_object_count > second_object_count

    def balance(self, objects, current_label):
        if current_label != 0:
            raise ValueError('Can only balance negative cases to positive ones for the time being')

        first_objects = objects[:, self.first_field_slice]
        second_objects = objects[:, self.second_field_slice]
        first_object_count = first_objects.eq(self.first_object_tensor).all(dim=1).sum()
        second_object_count = second_objects.eq(self.second_object_tensor).all(dim=1).sum()

        min_objects_to_modify = second_object_count - first_object_count + 1
        if second_object_count == objects.shape[0]:  # min_objects_to_modify > objects.shape[0]:
            second_object_indices = torch.nonzero(second_objects.eq(self.second_object_tensor).all(dim=1)).squeeze()
            num_second_objects_to_modify = random.randint(1, len(second_object_indices) - 1)
            second_object_indices_to_modify = second_object_indices[torch.randperm(len(second_object_indices))[:num_second_objects_to_modify]]

            new_assignment_options = list(range(self.second_field_gen.n_types))
            new_assignment_options.remove(self.second_field_index)
            new_assignments = [self.second_field_slice.start + c
                               for c in random.choices(new_assignment_options, k=num_second_objects_to_modify)]

            objects[second_object_indices_to_modify, self.second_field_slice.start + self.second_field_index] = 0
            objects[second_object_indices_to_modify, new_assignments] = 1
            min_objects_to_modify -= num_second_objects_to_modify

        if min_objects_to_modify <= 0:
            return objects

        max_objects_to_modify = objects.shape[0] - first_object_count
        num_objects_to_modify = random.randint(min_objects_to_modify, max_objects_to_modify)

        valid_indices_to_modify = torch.nonzero(~ (first_objects.eq(self.first_object_tensor).all(dim=1))).squeeze()
        if len(valid_indices_to_modify.shape) == 0:
            objects[valid_indices_to_modify, self.first_field_slice] = self.first_object_tensor
        else:
            indices_to_modify = valid_indices_to_modify[torch.randperm(valid_indices_to_modify.shape[0])][:num_objects_to_modify]
            objects[indices_to_modify, self.first_field_slice] = self.first_object_tensor

        return objects


class IdenticalObjectsRelation(ObjectRelation):
    def __init__(self, field_slices, field_generators, field_names=('color', 'shape'), position_field_names=('x', 'y')):
        super(IdenticalObjectsRelation, self).__init__(field_slices, field_generators, position_field_names)
        self.property_field_names = field_names
        self.property_field_slices = [self.field_slices[name] for name in self.property_field_names]
        self.property_field_generators = [self.field_generators[name] for name in self.property_field_names]

    def evaluate(self, objects):
        object_properties = torch.cat([objects[:, field_slice] for field_slice in self.property_field_slices],
                                      dim=1).to(torch.float).unsqueeze(0)

        for i in range(object_properties.shape[0] - 1):
            if (object_properties[i + 1:] == object_properties[i]).all(dim=1).any():
                return True

        return False

    def balance(self, objects, current_label):
        if current_label != 0:
            raise ValueError('Can only balance negative cases to positive ones for the time being')

        index_to_modify, index_to_copy_from = torch.randperm(objects.shape[0])[:2]

        for field_slice in self.property_field_slices:
            objects[index_to_modify, field_slice] = objects[index_to_copy_from, field_slice]

        return objects


class BetweenRelation(ObjectRelation):
    def __init__(self, field_slices, field_generators, relevant_field_name='color', outside_field_index=0,
                 inside_field_index=1, dtype=torch.float, position_field_names=('x', 'y')):
        super(BetweenRelation, self).__init__(field_slices, field_generators, position_field_names)
        self.dtype = dtype

        self.relevant_field_name = relevant_field_name
        self.relevant_field_slice = self.field_slices[self.relevant_field_name]
        self.relevant_field_gen = self.field_generators[self.relevant_field_name]

        self.outside_field_index = outside_field_index
        self.outside_object_tensor = torch.zeros(self.relevant_field_gen.n_types, dtype=self.dtype)
        self.outside_object_tensor[self.outside_field_index] = 1

        self.inside_field_index = inside_field_index
        self.inside_object_tensor = torch.zeros(self.relevant_field_gen.n_types, dtype=self.dtype)
        self.inside_object_tensor[self.inside_field_index] = 1

        self.between_objects_tensor = torch.stack((self.outside_object_tensor, self.inside_object_tensor, self.outside_object_tensor))

    def _sort_by_rows(self, objects, first_position_field=0):
        flattened_positions = objects.select(1, first_position_field) * self.position_field_generators[first_position_field].max_coord + objects.select(1, 1 - first_position_field)
        indices = flattened_positions.sort().indices
        return objects.index_select(0, indices)

    def _find_between_relation(self, objects, first_position_field=0):
        sorted_objects = self._sort_by_rows(objects, first_position_field)
        current_index = 0

        while current_index < sorted_objects.shape[0] - 2:
            if (sorted_objects[current_index, self.position_field_slices[first_position_field]] != sorted_objects[current_index + 1, self.position_field_slices[first_position_field]]) or \
               (sorted_objects[current_index, self.position_field_slices[1 - first_position_field]] + 1 != sorted_objects[current_index + 1, self.position_field_slices[1 - first_position_field]]):
                current_index += 1
                continue

            # At this point we know first two are in the same row and consecutive -- now check the second and third
            if (sorted_objects[current_index + 1, self.position_field_slices[first_position_field]] != sorted_objects[current_index + 2, self.position_field_slices[first_position_field]]) or \
               (sorted_objects[current_index + 1, self.position_field_slices[1 - first_position_field]] + 1 != sorted_objects[current_index + 2, self.position_field_slices[1 - first_position_field]]):
                # We know the next two are problematic, so we can skip checking them again
                current_index += 2
                continue

            #  We have three consecutive items -- so check if their values match
            if torch.all(sorted_objects[current_index:current_index + 3, self.relevant_field_slice] == self.between_objects_tensor):
                return True

            current_index += 1

        return False

    def evaluate(self, objects):
        return self._find_between_relation(objects, 0) or self._find_between_relation(objects, 1)

    def balance(self, objects, current_label):
        if current_label != 0:
            raise ValueError('Can only balance negative cases to positive ones for the time being')

        positions = torch.cat([objects[:, field_slice] for field_slice in self.position_field_slices],
                              dim=1).to(torch.float)

        # pick an object to utilize as a starting for the between triplet
        start_index = torch.randperm(objects.shape[0])[0]
        start_position = positions[start_index]

        # pick an axis
        same_axis = round(random.random())
        change_axis = 1 - same_axis

        direction = 0
        # pick a direction along the other axis
        if objects[start_index, self.position_field_slices[change_axis]] < self.position_field_generators[change_axis].min_coord + 2:
            direction = 1

        elif objects[start_index, self.position_field_slices[change_axis]] > self.position_field_generators[change_axis].max_coord - 3:
            direction = -1

        # the else condition, here because if torch.rand ~= 0, torch.sign will return 0, and will cause a bug
        while direction == 0:
            direction = int(torch.sign(torch.rand(tuple()) - 0.5))

        between_positions = torch.stack([start_position] * 3)
        between_positions[1, change_axis] += direction
        between_positions[2, change_axis] += 2 * direction

        indices_to_use = [start_index, None, None]
        valid_indices = list(range(objects.shape[0]))
        valid_indices.remove(start_index)

        # before we pick random indices to assign to these positions, make sure there are no objects there already
        for new_index in (1, 2):
            new_position = between_positions[new_index]
            matching_indices = (positions == new_position).all(dim=1).nonzero().squeeze()

            if matching_indices.nelement() > 0:
                if matching_indices.nelement() > 1:
                    # Should never happen
                    print(matching_indices)
                    print(positions)
                    print(objects)

                chosen_index = int(matching_indices)
                indices_to_use[new_index] = chosen_index
                valid_indices.remove(chosen_index)

        # now go through the new indices again, and if there weren't objects there assign random ones
        for new_index in (1, 2):
            if indices_to_use[new_index] is None:
                chosen_index = random.choice(valid_indices)
                indices_to_use.append(chosen_index)
                valid_indices.remove(chosen_index)

        for i in range(3):
            object_index = indices_to_use[i]
            for p in range(len(self.position_field_slices)):
                objects[object_index, self.position_field_slices[p]] = between_positions[i, p]

            objects[object_index, self.relevant_field_slice] = self.between_objects_tensor[i]

        return objects
