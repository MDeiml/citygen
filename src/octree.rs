use std::collections::{BTreeMap, HashMap};

pub fn build_octree(
    mut octree_buffer: glium::buffer::Mapping<[u32]>,
    voxels: glium::buffer::ReadMapping<[(u32, u32)]>,
    length: usize,
    depth: u32,
) -> usize {
    let mut child_map: HashMap<u32, [u32; 8]> = HashMap::new();
    let size: u32 = 1 << depth;
    for i in 0..length {
        let (pos, data) = voxels[i];
        let x = pos % size;
        let y = (pos / size) % size;
        let z = pos / size / size;
        let child_index = x % 2 + 2 * (y % 2 + 2 * (z % 2));
        let parent = (x / 2) + (size / 2) * ((y / 2) + (size / 2) * (z / 2));
        (*child_map.entry(parent).or_insert([0; 8]))[child_index as usize] = data;
    }

    let mut octree_buffer_index = 9; // TODO
    for d in (1..depth).rev() {
        let size: u32 = 1 << d;
        let mut voxel_map: BTreeMap<[u32; 8], Vec<u32>> = BTreeMap::new();
        for (parent, children) in child_map.drain() {
            voxel_map
                .entry(children)
                .or_insert_with(|| Vec::new())
                .push(parent);
        }
        for (children, parents) in voxel_map {
            let mut child_mask = 0xff00u32;
            let child_mask_index = octree_buffer_index;
            octree_buffer_index += 1;
            for i in 0..8 {
                if children[i] != 0 {
                    child_mask |= 1 << i;
                    octree_buffer[octree_buffer_index] = children[i];
                    octree_buffer_index += 1;
                }
            }
            octree_buffer[child_mask_index] = child_mask;
            for parent in parents {
                let parent_x = parent % size;
                let parent_y = (parent / size) % size;
                let parent_z = parent / size / size;
                let child_index = parent_x % 2 + 2 * (parent_y % 2 + 2 * (parent_z % 2));
                let parent_parent =
                    (parent_x / 2) + (size / 2) * ((parent_y / 2) + (size / 2) * (parent_z / 2));
                (*child_map.entry(parent_parent).or_insert([0; 8]))[child_index as usize] =
                    child_mask_index as u32;
            }
        }
    }
    let octree_length = octree_buffer_index;

    let children = child_map[&0];
    let mut child_mask = 0u32;
    octree_buffer_index = 1;
    for i in 0..8 {
        if children[i] != 0 {
            child_mask |= 1 << i;
            octree_buffer[octree_buffer_index] = children[i];
            octree_buffer_index += 1;
        }
    }
    octree_buffer[0] = child_mask;

    // for i in 0..octree_length {
    //     println!("{:06x}", octree_buffer[i]);
    // }
    println!("{}", octree_length);
    octree_length
}
