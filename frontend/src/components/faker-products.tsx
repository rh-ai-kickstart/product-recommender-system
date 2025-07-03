import { faker } from "@faker-js/faker";
import type { ProductData } from "../types";

export function FakerProducts() {
  function capitalizeFirstLetter(val: string) {
    return String(val).charAt(0).toUpperCase() + String(val).slice(1);
  }

  function createRandomProducts() {
    const myProduct: ProductData = {
      id: faker.number.int(),
      title: capitalizeFirstLetter(
        faker.word.noun({ length: { min: 5, max: 7 } })
      ),
      description: faker.lorem.lines(1),
      price: parseFloat(faker.number.float({ min: 5, max: 20, precision: 0.01 }).toFixed(2)),
      imageUrl: faker.image.urlLoremFlickr(),
      rating: parseFloat(faker.number.float({ min: 0, max: 5, precision: 0.1 }).toFixed(1)),
    };
    return myProduct;
  }

  const fakerData = faker.helpers.multiple(createRandomProducts, {
    count: 10,
  });

  return fakerData;
}
